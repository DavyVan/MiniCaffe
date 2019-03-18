/***
 * @file util.cpp
 * @author Quan Fan
 * @brief Implementation of util.h
 * @date 01/Mar/2019
 */

#include<cstring>
#include "util.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void alloc_and_strcpy(char** dst, const char* src)
{
    *dst = new char[strlen(src)];
    strcpy(*dst, src);
}

int random_matrix(float *A, int eleNum)
{
	srand(time(NULL));
	if (A == NULL) return -1;
	unsigned int i;
	for (i = 0; (A + i) && (i < eleNum); i++)
	{
		A[i] = (float)(rand() % 10) + 0.01 * (rand() % 100); 
	}

	if (i == eleNum) return 0;
	return -2;
}

int print_matrix(float *A, int batch_size, int x, int y, int z)
{
	if (A == NULL) return -1;
	float *batch_ptr = NULL;
	unsigned int i, j, k, curr_idx;
	
	for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
	{
		batch_ptr = A + curr_idx * x * y * z;
		printf("BATCH %d :\n", curr_idx);
		for (k = 0; k < z; k++)
		{
			printf("Layer %d\n", k);
			for (i = 0; i < y; i++)
			{	
				for (j = 0; (batch_ptr + (j * y + i) * x + k) && (j < x); j++)
				{
					printf("  %.2f  ", *(batch_ptr + (j * y + i) * x + k) );
				}
				printf("\n");
			}
		}
	}

	batch_ptr = NULL;

	if (k == z &&  i == y && j == x && curr_idx == batch_size) return 0;
	return -2;
}

int print_coord_matrix(coord_ptr A, int batch_size, int x, int y, int z)
{
	if (A == NULL) return -1;
	coord_ptr batch_ptr = NULL;
	unsigned int i, j, k;
	int curr_idx;
	for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
	{
		batch_ptr = A + curr_idx * x * y * z;
		printf("BATCH %d :\n", curr_idx);
		for (k = 0; k < z; k++)
		{
			printf("Layer %d\n", k);
			for (i = 0; i < y; i++)
			{	
				for (j = 0; (batch_ptr + (j * y + i) * x + k) && (j < x); j++)
				{
					printf("  (%d, %d, %d)  ", batch_ptr[(j * y + i) * x + k].col, batch_ptr[(j * y + i) * x + k].row, batch_ptr[(j * y + i) * x + k].z);
				}
				printf("\n");
			}
		}
	}

	batch_ptr = NULL;	

	if (k == z &&  i == y && j == x && curr_idx == batch_size) return 0;
	return -2;
}
