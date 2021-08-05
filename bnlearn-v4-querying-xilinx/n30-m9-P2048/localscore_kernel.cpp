#include <string.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <assert.h>
#include <ap_int.h>
#include <ap_fixed.h>

#define PE_NUM 2048
typedef ap_uint<32> varidx_t;
typedef ap_uint<32> varset_t;
typedef ap_uint<32> dataset_t;
typedef float score_t;

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
		#pragma HLS pipeline II=1
		a += COF[k] / (x + k);
	}
	return - x55 + (x + 0.5) * log(x55) + log(SQRT2PI * a / x);
}

extern "C" {

	void localscore(
		dataset_t *ptr_dataset,
		varidx_t *ptr_child_index,
		varset_t *ptr_parents_set,
		score_t *ptr_q,
		unsigned int num_of_dataset,
		unsigned int num_of_parent,
		unsigned int num_of_query
	){

		#pragma HLS INTERFACE s_axilite port=return bundle=control
		#pragma HLS INTERFACE s_axilite port=ptr_dataset bundle=control
		#pragma HLS INTERFACE s_axilite port=ptr_child_index bundle=control
		#pragma HLS INTERFACE s_axilite port=ptr_parents_set bundle=control
		#pragma HLS INTERFACE s_axilite port=ptr_q bundle=control
		#pragma HLS INTERFACE s_axilite port=num_of_dataset bundle=control
		#pragma HLS INTERFACE s_axilite port=num_of_parent bundle=control
		#pragma HLS INTERFACE s_axilite port=num_of_query bundle=control

		#pragma HLS INTERFACE m_axi port=ptr_dataset offset=slave bundle=gmem0
		#pragma HLS INTERFACE m_axi port=ptr_child_index offset=slave bundle=gmem1
		#pragma HLS INTERFACE m_axi port=ptr_parents_set offset=slave bundle=gmem1
		#pragma HLS INTERFACE m_axi port=ptr_q offset=slave bundle=gmem1

		varidx_t child_index[PE_NUM];
		varset_t parents_set[PE_NUM];
		score_t q[PE_NUM];
		score_t ls[PE_NUM];
		varset pattern[PE_NUM];
		unsigned int Nij0[PE_NUM];
		unsigned int Nij1[PE_NUM];

		unsigned int Nij0_cpy[PE_NUM];
		unsigned int Nij1_cpy[PE_NUM];

		// init
		unsigned int chunk_num = (num_of_query - 1)/PE_NUM + 1;

		// run
		assert(chunk_num <= (1<<26) / PE_NUM / std::max(std::max(sizeof(varidx_t), sizeof(varset_t)), sizeof(score_t)));
		chunk_loop:for (int chunk_index = 0; chunk_index < chunk_num; ++chunk_index){
			#pragma HLS loop_tripcount min=1 max=4096 avg=4096
			std::memcpy(child_index, ptr_child_index + chunk_index * PE_NUM, PE_NUM * sizeof(varidx_t));
			std::memcpy(parents_set, ptr_parents_set + chunk_index * PE_NUM, PE_NUM * sizeof(varset_t));

			pe_init_loop:for(int k = 0; k < PE_NUM; k++){
				#pragma HLS UNROLL

				ls[k] = 0;
				pattern[k] = 0;
				Nil0[k] = 0;
				Nij1[k] = 0;
			}

			assert(num_of_parent <= 8);
			parents_loop:for (int i = 0; i < (1<<num_of_parent); ++i){
				#pragma HLS loop_tripcount min=1 max=256 avg=256

				assert(num_of_dataset <= 10000);
				dataset_loop:for (unsigned int j = 0; j < num_of_dataset; ++j){
					#pragma HLS loop_tripcount min=1000 max=10000 avg=4500

					dataset_t data = ptr_dataset[j];
					pe_count_loop:for(int k = 0; k < PE_NUM; k++){
						#pragma HLS UNROLL

						if( (parents_set[k] & data) == pattern[k]){
							if ((data >> child_index[k]) & 1) Nij1[k]++;
							else Nij0[k]++;
						}
					}
				}

				std::memcpy(Nij0_cpy, Nij0, PE_NUM * sizeof(unsigned int));
				std::memcpy(Nij1_cpy, Nij1, PE_NUM * sizeof(unsigned int));

				pe_lgamma_loop:for(int k = 0; k < PE_NUM; k++){
					#pragma HLS PIPELINE II=1
					ls[k] += mylgamma(Nij0_cpy[k]+1) + mylgamma(Nij1_cpy[k]+1) - mylgamma(Nij0_cpy[k] + Nij1_cpy[k] + 2);
				}

				pe_update_loop:for(int k = 0; k < PE_NUM; k++){
					#pragma HLS UNROLL
					Nij0[k] = 0;
					Nij1[k] = 0;
					pattern[k] = ((pattern[k] | ~parents_set[k]) + 1) & parents_set[k];
				}
			}

			pe_term_loop:for(int k = 0; k < PE_NUM; k++){
				#pragma HLS PIPELINE II=1
				ptr_q[chunk_index * PE_NUM + k] = ls[k];
			}
		}
	}
}
