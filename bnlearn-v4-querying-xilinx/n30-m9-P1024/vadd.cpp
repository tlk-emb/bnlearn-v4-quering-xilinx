#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "vadd.h"
#include <fstream>
#include <string>
#include <queue>
#include <tuple>


#define PE_NUM 1024
#define NUMOF_DATASETS 10000
#define MAXOF_VARS 30

typedef uint32_t varidx_t;
typedef uint32_t varset_t;
typedef uint32_t dataset_t;
typedef float score_t;

static const bool CALC_LS_TEST = false;
static const int MAXOF_PARENTS = 9;
static const int N = 30;
//static const int N = 10;
static const int CACHE_SIZE = 28;
//static const int CACHE_SIZE = 8;
static const score_t SCORE_INF = -(1<<30);

const char* dataset_filepath = "../../src/dataset/asia30.idt";
// char* dataset_filepath = "../alarm.idt";

int active_num;

cl_ulong a,b;
std::queue<std::pair<int, varset_t>> q_varset[2];
std::queue<std::pair<varidx_t, varset_t>> q_queries;

long long _nCk[(N+1)*(N+2)/2] = {1}; // {1,0,0,0, ... }
long long nCk(int n, int k){
    k = std::min(k, n-k);
    if (n < 0 || k < 0 || n < k) return 0;
    if (_nCk[n*(n+1)/2+k] != 0) return _nCk[n*(n+1)/2+k];
    return _nCk[n*(n+1)/2+k] = nCk(n-1, k) + nCk(n-1, k-1);
}
int *index_memo;
void init_index(int cache_size){
  std::queue<std::pair<int,int>> q[2];
  // first step
  q[0].push(std::make_pair<int,int>(0, 0));
  index_memo[0] = 0;

  for (int stage = 1; stage <= cache_size; ++stage){

    int index = 0;
    while(!q[(stage-1)%2].empty()){
      auto p = q[(stage-1)%2].front(); q[(stage-1)%2].pop();
      int i = p.first;
      int varset = p.second;

      for ( ; i < cache_size; ++i){
        int new_varset = varset ^ (1<<i);
        index_memo[new_varset] = index++;
        q[stage%2].push(std::make_pair(i+1, new_varset));
      }
    }
  }
}
int varset2index(int n, int k, varset_t S){
  if (n == CACHE_SIZE) return index_memo[S];
  if (S == 0) return 0;
  if (S & 1) return varset2index(n-1, k-1, S>>1);
  return nCk(n-1, k-1) + varset2index(n-1, k, S>>1);
}
void load_data(int N, dataset_t data[NUMOF_DATASETS], std::string filepath){
	std::ifstream ifs;
	ifs.open(filepath, std::ios::in);
	if (!ifs) {
	  std::cerr << "training file open failed" << std::endl;
	}
	int i = 0;
	while(!ifs.eof() && i < NUMOF_DATASETS){
		std::string line;
		getline(ifs, line);
		if(line == "") break;
		int pos = 0;
		uint32_t d = 0;
		for(int j = 0; j < (int)line.size(); j++){
			if(pos >= N) break;
			if(line[j] != ' '){
				unsigned int tmp_d = ((unsigned int)(line[j] - '0'));
				if(tmp_d > 3) std::cerr << "Warning: the value of dataset must be less than 4. line = " <<  i << "tmp_d = " << tmp_d << std::endl;
				d |= (tmp_d << pos);
				pos++;
			}
		}
		data[i] = d;
		i++;
	}
	ifs.close();
}

//void on_return(cl_event event, cl_int exec_status, void* arg)
//{
//   std::cout << "device ends ... :" << active_pe_num << std::endl;
//
//  int ls_index = 0;
//  while (ls_index < active_pe_num){
//    q_answers.push(std::make_tuple(ptr_child_index[ls_index], ptr_parents_set[ls_index], ptr_q[ls_index]));
//    ls_index++;
//  }
//  device_status = 0;
//}


int main(int argc, char* argv[]) {
	std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    //TARGET_DEVICE macro needs to be passed from gcc command line
    if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}

    char* xclbinFilename = argv[1];
    
    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary
    
    std::vector<cl::Device> devices;
    cl::Device device;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    //traversing all Platforms To find Xilinx Platform and targeted
    //Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if ( platformName == "Xilinx"){
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	    if (devices.size()){
		    device = devices[0];
		    found_device = true;
		    break;
	    }
        }
    }
    if (found_device == false){
       std::cout << "Error: Unable to find Target Device " 
           << device.getInfo<CL_DEVICE_NAME>() << std::endl;
       return EXIT_FAILURE; 
    }

    // Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load xclbin 
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    cl::Program program(context, devices, bins);
    
    // This call will get the kernel object from program. A kernel is an 
    // OpenCL function that is executed on the FPGA. 
    cl::Kernel kernel(program,"localscore");
    
    int chunk_num = (1<<26) / PE_NUM / std::max(std::max(sizeof(varidx_t), sizeof(varset_t)), sizeof(score_t));
    std::cout << "chunk_num : " << chunk_num << std::endl;

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device. 
	cl::Buffer buffer_dataset(context, CL_MEM_READ_ONLY, NUMOF_DATASETS * sizeof(dataset_t));
    cl::Buffer buffer_child_index(context, CL_MEM_READ_ONLY, PE_NUM * chunk_num * sizeof(varidx_t));
    cl::Buffer buffer_parents_set(context, CL_MEM_READ_ONLY, PE_NUM * chunk_num * sizeof(varset_t));
    cl::Buffer buffer_q(context, CL_MEM_WRITE_ONLY, PE_NUM * chunk_num * sizeof(score_t));
    
    //set the kernel Arguments
    int narg=0;
    kernel.setArg(narg++,buffer_dataset);
    kernel.setArg(narg++,buffer_child_index);
    kernel.setArg(narg++,buffer_parents_set);
    kernel.setArg(narg++,buffer_q);
    kernel.setArg(narg++,NUMOF_DATASETS);

    //We then need to map our OpenCL buffers to get the pointers
    dataset_t *ptr_dataset = (dataset_t *) queue.enqueueMapBuffer (buffer_dataset, CL_TRUE , CL_MAP_WRITE , 0, NUMOF_DATASETS * sizeof(dataset_t));
    varidx_t *ptr_child_index = (varidx_t *) queue.enqueueMapBuffer (buffer_child_index, CL_TRUE , CL_MAP_WRITE , 0, PE_NUM * chunk_num * sizeof(varidx_t));
    varset_t *ptr_parents_set = (varset_t *) queue.enqueueMapBuffer (buffer_parents_set,	CL_TRUE , CL_MAP_WRITE , 0, PE_NUM * chunk_num * sizeof(varset_t));
    score_t *ptr_q = (score_t *) queue.enqueueMapBuffer (buffer_q, CL_TRUE , CL_MAP_READ  , 0, PE_NUM * chunk_num * sizeof(score_t));


    std::cout << "loding data ..." << std::endl;
    load_data(N, ptr_dataset, dataset_filepath);

    // Data will be migrated to kernel space
    cl::Event event_init;
    queue.enqueueMigrateMemObjects({buffer_dataset}, 0, NULL, &event_init);
    cl::Event::waitForEvents({event_init});

    std::cout << "Kernel initialization finished" << std::endl;
    std::cout << std::endl;

    // solve
    std::cout << "start calc ..." << std::endl;

    long long memory_sum = 0;
    std::cout << "N : " << N << " parent limit : " << MAXOF_PARENTS << " dataset size : " << NUMOF_DATASETS << std::endl;


    // index memo
    index_memo = new int[(1<<CACHE_SIZE)];
    memory_sum += sizeof(int)*(1<<CACHE_SIZE);
    std::cout << "index_memo size : " << sizeof(int)*(1<<CACHE_SIZE) / 1000 << "[KB]" << std::endl;
    init_index(CACHE_SIZE);

    // best local score
    score_t ***best_local_score = new score_t**[2];
    for (int i = 0; i < 2; ++i){
      best_local_score[i] = new score_t*[N];
      for (int j = 0; j < N; ++j){
        // todo
        best_local_score[i][j] = new score_t[nCk(N-1, (N-1)/2)];
      }
    }
    memory_sum += sizeof(score_t)*2*N*nCk(N-1, (N-1)/2);
    std::cout << "best_local_score size : " << sizeof(score_t)*2*N*nCk(N-1, (N-1)/2) / 1000 << "[KB]" << std::endl;


    // best glaph score
    score_t **best_glaph_score = new score_t*[2];
    for (int i = 0; i < 2; ++i){
      best_glaph_score[i] = new score_t[nCk(N, N/2)];
    }
    memory_sum += sizeof(score_t)*2*nCk(N, N/2);
    std::cout << "best_glaph_score size : " << sizeof(score_t)*2*nCk(N, N/2) / 1000 << "[KB]" << std::endl;

    best_glaph_score[0][0] = 0;
    q_varset[0].push(std::make_pair(0, 0));
    for (int stage = 1; stage <= N; ++stage){
      std::cout << "stage:" << stage << " starts ..." << std::endl;

      for (int i = 0; i < nCk(N, N/2); ++i){
        best_glaph_score[stage%2][i] = SCORE_INF;
      }

      if (stage-1 > MAXOF_PARENTS){
//    	  break;
          while(!q_varset[(stage-1)%2].empty()){
            auto p = q_varset[(stage-1)%2].front(); q_varset[(stage-1)%2].pop();
            int cursor = p.first;
            varset_t v = p.second;

            for (int i = 0; i < N; ++i){
              if ( !((1 << i) & v) ){

				varidx_t child_index = i;
				varset_t parents_set = v;
				score_t local_score = SCORE_INF;
				// std::cout << "integrate child:" << child_index << " parents:" << parents_set << " score:" << local_score << " starts ... " << std::endl;

				varset_t digit = 1<<child_index;                // ...000001000
				varset_t lower_bitmask = digit-1;               // ...000000111
				varset_t upper_bitmask = -digit;                // ...111111000
				if(stage > 1){
				  for (int i = 0; i < N; ++i){
					if  (parents_set & (1 << i)){
					  varset_t p_comp = parents_set ^ (1<<i);
					  p_comp = ((p_comp & upper_bitmask) >> 1) | (p_comp & lower_bitmask);
					  local_score = std::max(
						local_score,
						best_local_score[(stage-1)%2][child_index][varset2index(N-1, stage-2, p_comp)]
					  );
					}
				  }
				}
				varset_t p_comp = parents_set;
				p_comp = ((p_comp & upper_bitmask) >> 1) | (p_comp & lower_bitmask);
				best_local_score[stage%2][child_index][varset2index(N-1, stage-1, p_comp)] = local_score;

				best_glaph_score[stage%2][varset2index(N, stage, parents_set ^ (1 << child_index))] = std::max(
				  best_glaph_score[stage%2][varset2index(N, stage, parents_set ^ (1 << child_index))],
				  best_glaph_score[(stage-1)%2][varset2index(N, stage-1, parents_set)] + local_score
				);

              }
            }

            // next varset
            for (; cursor < N; ++cursor){
              q_varset[stage%2].push(std::make_pair(cursor+1, v ^ (1 << cursor)) );
            }
          }
      }else{
          while(!q_varset[(stage-1)%2].empty()){
            auto p = q_varset[(stage-1)%2].front(); q_varset[(stage-1)%2].pop();
            int cursor = p.first;
            varset_t v = p.second;

            for (int i = 0; i < N; ++i){
              if ( !((1 << i) & v) ){
            	  q_queries.push(std::make_pair(i, v));
              }
            }

            // next varset
            for (; cursor < N; ++cursor){
              q_varset[stage%2].push(std::make_pair(cursor+1, v ^ (1 << cursor)) );
            }
          }

          std::cout << "stage:" << stage << " pushed queries " << std::endl;


    	  int num = q_queries.size();
    	  std::chrono::system_clock::time_point calc_ls_start = std::chrono::system_clock::now();
		  while(!q_queries.empty()){
        	  int ls_index = 0;
			  while (ls_index < PE_NUM * chunk_num && !q_queries.empty()){
				auto query = q_queries.front(); q_queries.pop();
				ptr_child_index[ls_index] = query.first;
				ptr_parents_set[ls_index] = query.second;
				ls_index++;
			  }
			  active_num = ls_index;
			  kernel.setArg(narg, stage-1);
			  kernel.setArg(narg+1, active_num);

			  cl::Event event_write;
			  cl::Event event_run;
			  cl::Event event_read;

			  queue.enqueueMigrateMemObjects({buffer_child_index, buffer_parents_set}, 0, NULL, &event_write);
			  queue.enqueueTask(kernel, NULL, &event_run);
			  queue.enqueueMigrateMemObjects({buffer_q}, CL_MIGRATE_MEM_OBJECT_HOST, NULL, &event_read);
			  std::cout << "device started ... :" << num - q_queries.size() << "/" << num << std::endl;
			  std::chrono::system_clock::time_point device_start = std::chrono::system_clock::now();
			  queue.finish();
			  std::chrono::system_clock::time_point device_end = std::chrono::system_clock::now();
			  std::cout << "device ended ... " << std::endl;
			  double elapsed2 = (double)std::chrono::duration_cast<std::chrono::microseconds>(device_end-device_start).count();
			  std::cout << std::fixed << std::setprecision(6) << "device calculated " << active_num << " local scores in " << elapsed2 << "[us] = " << elapsed2/1000 << "[ms] = " << elapsed2 / 1000000 << "[s] (" << (double)active_num * 1000000 / elapsed2 << ")" << std::endl;

			  ls_index = 0;
			  while (!CALC_LS_TEST && ls_index < active_num){

				  varidx_t child_index = ptr_child_index[ls_index];
				  varset_t parents_set = ptr_parents_set[ls_index];
				  score_t local_score = ptr_q[ls_index];
				  // std::cout << "integrate child:" << child_index << " parents:" << parents_set << " score:" << local_score << " starts ... " << std::endl;

				  varset_t digit = 1<<child_index;                // ...000001000
				  varset_t lower_bitmask = digit-1;               // ...000000111
				  varset_t upper_bitmask = -digit;                // ...111111000
				  if(stage > 1){
					for (int i = 0; i < N; ++i){
					  if  (parents_set & (1 << i)){
						varset_t p_comp = parents_set ^ (1<<i);
						p_comp = ((p_comp & upper_bitmask) >> 1) | (p_comp & lower_bitmask);
						local_score = std::max(
						  local_score,
						  best_local_score[(stage-1)%2][child_index][varset2index(N-1, stage-2, p_comp)]
						);
					  }
					}
				  }
				  varset_t p_comp = parents_set;
				  p_comp = ((p_comp & upper_bitmask) >> 1) | (p_comp & lower_bitmask);
				  best_local_score[stage%2][child_index][varset2index(N-1, stage-1, p_comp)] = local_score;

				  best_glaph_score[stage%2][varset2index(N, stage, parents_set ^ (1 << child_index))] = std::max(
					best_glaph_score[stage%2][varset2index(N, stage, parents_set ^ (1 << child_index))],
					best_glaph_score[(stage-1)%2][varset2index(N, stage-1, parents_set)] + local_score
				  );

				  ls_index++;
			  }
          }
		  std::chrono::system_clock::time_point calc_ls_end = std::chrono::system_clock::now();
		  double elapsed3 = (double)std::chrono::duration_cast<std::chrono::microseconds>(calc_ls_end-calc_ls_start).count();
		  if (CALC_LS_TEST) std::cout << "num of parents:" << stage-1 <<  " device calculated " << num << " local scores in " << elapsed3 << "[us] = " << elapsed3/1000 << "[ms] = " << elapsed3/1000000 << "[s]" << std::endl;
      }
    }


    std::cout << "Kernel execution is complete." << std::endl;

    // print result
    std::cout << "best_score : " << best_glaph_score[N%2][0] << std::endl;
    // std::cout << "best_matrix" << std::endl;

    queue.enqueueUnmapMemObject(buffer_dataset, ptr_dataset);
    queue.enqueueUnmapMemObject(buffer_child_index, ptr_child_index);
    queue.enqueueUnmapMemObject(buffer_parents_set, ptr_parents_set);
    queue.enqueueUnmapMemObject(buffer_q, ptr_q);
    queue.finish();

    // print total time
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    double elapsed1 = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << "ALL TIME : " << elapsed1 << "[ms] = " << elapsed1 / 1000 << "[s]" << std::endl;
    std::cout << "rough estimate of MEMORY : " << memory_sum / 1000 << "[KB] = " << memory_sum / 1000000 << "[MB] = " << memory_sum / 1000000000 << "[GB]" << std::endl;

}
