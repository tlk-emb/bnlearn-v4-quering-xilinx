#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <queue>
#include <tuple>
#include <chrono>

#define NUMOF_DATASETS 10000
#define MAXOF_VARS 30

typedef uint32_t varidx_t;
typedef uint32_t varset_t;
typedef uint32_t dataset_t;
typedef float score_t;

static const bool CALC_LS_TEST = false;
static const int MAXOF_PARENTS = 6;
static const int N = 30;
// static const int N = 10;
static const int CACHE_SIZE = 28;
// static const int CACHE_SIZE = 8;
static const score_t SCORE_INF = -(1<<30);

const char* dataset_filepath = "../../src/dataset/asia30.idt";
// char* dataset_filepath = "../alarm.idt";

std::queue<std::pair<int, varset_t>> q_varset[2];
std::queue<std::pair<varidx_t, varset_t>> q_queries;

score_t mylgamma(score_t x){
	score_t SQRT2PI = 2.5066282746310005;
	score_t COF[7] = {
	  1.000000000190015, 76.18009172947146, -86.50532032941677,
	  24.01409824083091, -1.231739572450155, 0.1208650973866179e-2,
	  -0.5395239384953e-5
	};
	score_t x55 = x + 5.5;
	score_t a = COF[0];

	for(int k = 1; k < 7; ++k){
		a += COF[k] / (x + k);
	}
	return - x55 + (x + 0.5) * log(x55) + log(SQRT2PI * a / x);
}


score_t PE(
		dataset_t dataset[NUMOF_DATASETS],
		varidx_t child_index,
		varset_t parents_set,
		unsigned int num_of_dataset,
		unsigned int num_of_parent
){
    score_t ls = 0;
    uint32_t lowest_bit = parents_set & (-parents_set);
    uint32_t pattern = 0;
    for (int i = 0; i < (1<<num_of_parent); ++i){
		unsigned int Nij0 = 0, Nij1 = 0;
        for (unsigned int j = 0; j < num_of_dataset; ++j){
			uint32_t data = dataset[j];
            if( (parents_set & data) == pattern){
                if ((data >> child_index) & 1) Nij1++;
                else Nij0++;
            }
        }
        ls += mylgamma(Nij0+1) + mylgamma(Nij1+1) - mylgamma(Nij0 + Nij1 + 2);
        pattern = ((pattern | ~parents_set) + lowest_bit) & parents_set;
    }
    return ls;
}

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

int main(int argc, char* argv[]) {
	std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();

    dataset_t dataset[NUMOF_DATASETS];
    std::cout << "loding data ..." << std::endl;
    load_data(N, dataset, dataset_filepath);


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

    	  std::chrono::system_clock::time_point calc_ls_start = std::chrono::system_clock::now();
		  while(!q_queries.empty()){
			auto query = q_queries.front(); q_queries.pop();
			auto child_index = query.first;
			auto parents_set = query.second;
			score_t local_score = PE(dataset, child_index, parents_set, NUMOF_DATASETS, stage-1);
			// std::cout << "integrate child:" << child_index << " parents:" << parents_set << " score:" << local_score << " starts ... " << std::endl;

			if(CALC_LS_TEST) continue;
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
		  std::chrono::system_clock::time_point calc_ls_end = std::chrono::system_clock::now();
		  double elapsed3 = (double)std::chrono::duration_cast<std::chrono::microseconds>(calc_ls_end-calc_ls_start).count();
		  if (CALC_LS_TEST) std::cout << std::fixed << std::setprecision(6) << "num of parents:" << stage-1 << " local scores in " << elapsed3 << "[us] = " << elapsed3/1000 << "[ms] = " << elapsed3/1000000 << "[s]" << std::endl;
      }
    }


    std::cout << "Kernel execution is complete." << std::endl;

    // print result
    std::cout << "best_score : " << best_glaph_score[N%2][0] << std::endl;
    // std::cout << "best_matrix" << std::endl;

    // print total time
    std::chrono::system_clock::time_point t2 = std::chrono::system_clock::now();
    double elapsed1 = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    std::cout << "ALL TIME : " << elapsed1 << "[ms] = " << elapsed1 / 1000 << "[s]" << std::endl;
    std::cout << "rough estimate of MEMORY : " << memory_sum / 1000 << "[KB] = " << memory_sum / 1000000 << "[MB] = " << memory_sum / 1000000000 << "[GB]" << std::endl;

}
