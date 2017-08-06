#include<stdio.h>
#include <unistd.h>
#include<cuda.h>
#include<math.h>
#include <stdlib.h>
#include <time.h>
#include "/usr/local/cuda/samples/common/book.h"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#define cities 20
#define chromosomes 20
__global__ void assignC(int *solutions);
__global__ void setRandomPath(int * chromosome,double *r);
__global__ void totalDistance(int *chromosome,double *distances,double *x,double *y);
__global__ void evaluateFitness(double *distances,double *fit);
__host__ __device__ double* maxelement ( double* first, double* last );
__device__ int* rouletteSelection(double *fit,int *sol,double *rf,double sum);
__device__ void mutate(int *chromosome,double mutprob,double *r1,double *rc);
__device__ void crossover(int *parentA,int *parentB,int *offspringA,int* offspringB,double crossprob,double *r1,double *rc);
__device__ int* findI(int* first, int* last, int val);
__device__ void repairOffspring(int *offspringToRepair, int missingIndex,int* other);
__device__ bool hasDuplicate(int *chromosome,int popCount,int *newpop);


__global__ void assignC(int *solutions)
{
	int i=threadIdx.y,j=threadIdx.x;
	*(solutions + i*chromosomes + j)=j;

}
__global__ void prinT(int *chromosome)
{
	int i=threadIdx.y,j=threadIdx.x;
	printf("sol[%d][%d]=%d\n",i,j,*(chromosome+i*chromosomes+j));
}
__global__ void printchrom(int *chromosome)
{
	int i,j;
	for(i=0;i<chromosomes;i++)
	{
		for(j=0;j<cities;j++)
		{
			printf("solutionsc[%d][%d]=%d\t",i,j,*((chromosome+i*chromosomes)+j));
		}
		printf("\n");
	}


}
__global__ void setRandomPath(int * chromosome,double *r)
{

		int i;
		for(i=0;i<cities;i++)
		{
			int random = (int)(*(r+i)*cities);
			int temp = *(chromosome+i);
			*(chromosome+i)= *(chromosome+random);
			*(chromosome+random) = temp;
		}
}
__global__ void totalDistance(int *chromosome,double *distances,double *x,double *y)
{
	int i=threadIdx.x,j=0;
	/* Calculate the total distance between all cities */
	//printf("sol[%d][%d]=%d\n",i,j,*(chromosome+j));
	//printf("citiesx[%d]=%lf,citiesY[%d]=%lf\n",j,x[j],j,y[j]);
	for(j=0;j<chromosomes;j++)
	{
		if(i<cities-1)
		{
			double dx = x[*(chromosome+j*chromosomes+i)] - x[*(chromosome+j*chromosomes+i+1)];
			double dy = y[*(chromosome+j*chromosomes+i)] - y[*(chromosome+j*chromosomes+i+1)];
			*(distances+j*chromosomes+i)= sqrt((pow(dx, 2.0) + pow(dy, 2.0)));
			//printf("1distance %d  %lf\n",i,distances[i]);
			/* We complete the tour by adding the distance between the last and the first city */
			//printf("2distance %d  %lf\n",i,distance);
		}
		__syncthreads();
		if(i>=cities-1 && i<cities)
		{
			double dx = x[*(chromosome+j*chromosomes+cities-1)] - x[*chromosome];
			double dy = y[*(chromosome+j*chromosomes+cities-1)] - y[*chromosome];
			*(distances+j*chromosomes+i)= sqrt((pow(dx, 2.0) + pow(dy, 2.0)));
		}
	}
}
__global__ void evaluateFitness(double *distances,double *fit)
{
	int i,j=0;
	for(j=0;j<chromosomes;j++)
	{
		double dist=0;
		for(i=0;i<cities;i++)
		 dist+=*(distances+j*chromosomes+i);
		*(fit+j)=1/dist;
	}
}
__host__ __device__ double* maxelement ( double* first, double* last )
{
	if (first==last) return last;
	 double* largest = first,*i=first;

	 while (i!=last)
	 {
		 i=i+1;
		if (*(largest)<*(i))    // or: if (comp(*largest,*first)) for version (2)
	     largest=i;
	  }
	 return largest;
}

__device__ int* rouletteSelection(double *fit,int *sol,double *rf,double s)
{
	double sum=0,ran;int i;
	ran=(*rf)*s;
	for(i=0;i<chromosomes;i++)
	{
		sum+=*(fit+i);
		if(sum >= ran)
		{
			return (sol+i*chromosomes);
		}
	}
	return NULL;
}
__device__ void mutate(int *chromosome,double mutprob,double *r1,double *rc)
{
	int ran1,ran2,tmp;
	if(*r1 > mutprob)
		return;
	else
	{
		ran1=(int)(*rc*cities);
		ran2=(int)(*(rc+1)*cities);
		if(ran1==ran2)
		ran2=(int)(*(rc+2)*cities);
		tmp = *(chromosome+ran1);
		*(chromosome+ran1) = *(chromosome+ran2);
		*(chromosome+ran2) = tmp;
	}
}
__device__ void crossover(int *parentA,int *parentB,int *offspringA,int* offspringB,double crossprob,double *r1,double *rc)
{
	int i,j;
	if(*r1 >crossprob)
	{
		memcpy(offspringA,parentA,sizeof(int)*cities);
		memcpy(offspringB,parentB,sizeof(int)*cities);
		return;
	}
	else
	{
		int cutoffindex1=(int)(*rc * cities);
		int cutoffindex2=(int)(*(rc+1) * cities);
		if(cutoffindex1==cutoffindex2)
			cutoffindex2=(int)(*(rc+2) * cities);
		int start,end;
		if(cutoffindex1<cutoffindex2)
		{
			start=cutoffindex1;end=cutoffindex2;
		}
		else
		{
			start=cutoffindex2;end=cutoffindex1;
		}
		memcpy(offspringA, parentA, sizeof(int) * cities);
		memcpy(offspringB, parentB, sizeof(int) * cities);
		memcpy(offspringA + start, parentB + start, sizeof(int) * (end - start));
		memcpy(offspringB + start, parentA + start, sizeof(int) * (end - start));
		for(i=0;i<cities;i++)
		{
			if(i>=start && i<end)
			{

			}
			else
			{
				for(j=start;j<end;j++)
				{
					if(offspringA[i]==offspringA[j])
						offspringA[i]=-1;
					if(offspringB[i]==offspringB[j])
						offspringB[i]=-1;
				}
			}
		}
		for(i=0;i<cities;i++)
		{
			if(offspringA[i] == -1)
			{
				repairOffspring(offspringA, i, offspringB);
			}
			if(offspringB[i] == -1)
			{
				repairOffspring(offspringB, i, offspringA);
			}
		}
	}
}
__device__ int* findI(int* first, int* last, int val)
{
  while (first!=last) {
    if (*first==val) return first;
    ++first;
  }
  return last;
}
__device__ void repairOffspring(int *offspringToRepair, int missingIndex,int* other)
{
	int i;
	for(i=0;i<cities;i++)
	{
		int *missing=findI(offspringToRepair, offspringToRepair + cities, other[i]);
		if(missing == (offspringToRepair + cities))
		{
			offspringToRepair[missingIndex] = other[i];
			return;
		}
	}
}
__device__ bool hasDuplicate(int *chromosome,int popCount,int *newpop)
{
	int i,gene;
	for(i=0;i<popCount;i++)
	{
		int genescompared=0;
		for(gene=0;gene<cities;gene++)
		{
			if(*(chromosome+gene)!=*(newpop+i*chromosomes+gene))
			{
				break;
			}
			++genescompared;
		}
		if(genescompared==cities)
			return true;
	}
	return false;
}
__device__ void copyToNewPopulation(int *chromosome,int *newpop,int index)
{
	int i=threadIdx.x;
	if(i<cities)
	{
		*(newpop+index*chromosomes+i)=*(chromosome+i);
	}
}
__global__ void nextPopulation(int *sol,int *newpop,int *best,double *fit,double crossprob,double mutprob,double *rf1,double *rf2,double *r1,double *rc,double s)
{
	int eliteindex1=0,eliteindex2=0,i;
	eliteindex1=(maxelement(fit,fit+chromosomes) - fit);
	//cudaMemcpy(best,sol+eliteindex1*chromosomes,sizeof(int)*cities,cudaMemcpyDeviceToDevice);
	copyToNewPopulation(sol+eliteindex1*chromosomes,best,0);
	double highestFitness = 0;
	for(i=0;i<chromosomes;i++)
	{
		if(i!=eliteindex1 && *(fit+i)>highestFitness)
		{
			highestFitness=*(fit+i);
			eliteindex2=i;
		}
	}
	int offspringcount=0;
	//cudaMemcpy(newpop+offspringcount*chromosomes,sol+eliteindex1*chromosomes,cities*sizeof(int),cudaMemcpyDeviceToDevice);
	//memcpy(newpop+offspringcount*chromosomes,sol+eliteindex1*chromosomes,cities*sizeof(int));
	copyToNewPopulation(sol+eliteindex1*chromosomes,newpop,offspringcount);
	offspringcount++;
	//cudaMemcpy(newpop+offspringcount*chromosomes,sol+eliteindex2*chromosomes,cities*sizeof(int),cudaMemcpyDeviceToDevice);
	//memcpy(newpop+offspringcount*chromosomes,sol+eliteindex2*chromosomes,cities*sizeof(int));
	copyToNewPopulation(sol+eliteindex2*chromosomes,newpop,offspringcount);
	offspringcount++;
	//while(offspringcount!=chromosomes)
	//{
		int *pA,*pB;
		pA=rouletteSelection(fit,sol,rf1,s);
		pB=rouletteSelection(fit,sol,rf2,s);
		int offspringA[cities];
		int offspringB[cities];
		crossover(pA, pB, offspringA, offspringB,crossprob,r1,rc);
		mutate(offspringA,mutprob,r1,rc);
		mutate(offspringB,mutprob,r1,rc);
		if(!hasDuplicate(offspringA, offspringcount,newpop))
		{
			//cudaMemcpy(newpop+offspringcount*chromosomes,offspringA,cities*sizeof(int),cudaMemcpyDeviceToDevice);
			copyToNewPopulation(offspringA,newpop,offspringcount);
			++offspringcount;
		}
		if(offspringcount == chromosomes)
		{
			return;
		}
		if(!hasDuplicate(offspringB, offspringcount,newpop))
		{
			//cudaMemcpy(newpop+offspringcount*chromosomes,offspringB,cities*sizeof(int),cudaMemcpyDeviceToDevice);
			copyToNewPopulation(offspringB, newpop,offspringcount);
			++offspringcount;
		}
		if(offspringcount == chromosomes)
		{
			return;
		}
	//}
	memcpy(sol,newpop,chromosomes*cities*sizeof(int));

}
int main()
{
	curandGenerator_t gen;
	int i,generations=0,generationsWithoutImprovement=0,maxFitPos;
	double crossoverProbability=0.9,mutationProbability=0.2,newFit,bestFitness=-1;
	double sum=0,*dev_rf1,*dev_rf2,*dev_r1,*dev_rc;
	double *dev_x,*dev_y,*dev_distances,distances,*citiesX,*citiesY,*fitness,*dev_fitness;
	int *dev_solutions,*dev_newPopulation,*bestchromosome,*dev_bestchromosome;
	bestchromosome=(int*)malloc(cities*sizeof(int));
	citiesX=(double*)malloc(cities*sizeof(double));
	citiesY=(double*)malloc(cities*sizeof(double));
	fitness=(double*)malloc(chromosomes*sizeof(double));
	citiesX[0] =60 ;citiesY[0] =200 ;
	citiesX[1] =180 ;citiesY[1] =200 ;
	citiesX[2] =80 ;citiesY[2] =180 ;
	citiesX[3] =140 ;citiesY[3] =180 ;
	citiesX[4] =20 ;citiesY[4] =160 ;
	citiesX[5] =100 ;citiesY[5] =160 ;
	citiesX[6] =200 ;citiesY[6] =160 ;
	citiesX[7] =140 ;citiesY[7] =140 ;
	citiesX[8] =40 ;citiesY[8] =120 ;
	citiesX[9] =100 ;citiesY[9] =120 ;
	citiesX[10] =180 ;citiesY[10] =100 ;
	citiesX[11] =60 ;citiesY[11] =80 ;
	citiesX[12] =120 ;citiesY[12] =80 ;
	citiesX[13] =180 ;citiesY[13] =60 ;
	citiesX[14] =20 ;citiesY[14] =40 ;
	citiesX[15] =100 ;citiesY[15] =40 ;
	citiesX[16] =200 ;citiesY[16] =40 ;
	citiesX[17] =20 ;citiesY[17] =20 ;
	citiesX[18] =60 ;citiesY[18] =20 ;
	citiesX[19] =160 ;citiesY[19] =20 ;
	cudaMalloc((void**)&dev_x,cities*sizeof(double));
	cudaMalloc((void**)&dev_y,cities*sizeof(double));
	cudaMalloc((void**)&dev_fitness,chromosomes*sizeof(double));
	cudaMalloc((void**)&dev_distances,chromosomes*cities*sizeof(double));
	cudaMalloc((void**)&dev_solutions,chromosomes*cities*sizeof(int));
	cudaMalloc((void**)&dev_newPopulation,chromosomes*cities*sizeof(int));
	cudaMalloc((void**)&dev_bestchromosome,cities*sizeof(int));
	cudaMalloc((void**)&dev_rf1,chromosomes*sizeof(double));
	cudaMalloc((void**)&dev_rf2,chromosomes*sizeof(double));
	cudaMalloc((void**)&dev_r1,chromosomes*sizeof(double));
	cudaMalloc((void**)&dev_rc,chromosomes*sizeof(double));
	cudaMemcpy(dev_x,citiesX,cities*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y,citiesY,cities*sizeof(double),cudaMemcpyHostToDevice);
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
	dim3 grid(1,1,1);
	dim3 block(chromosomes,cities,1);
	assignC<<<grid,block>>>(dev_solutions);
	for(i=0;i<chromosomes;i++)
	{
		curandGenerateUniformDouble(gen, dev_rc, cities);
		setRandomPath<<<1,1>>>(dev_solutions+i*chromosomes,dev_rc);
	}
	//while(generationsWithoutImprovement<100)
	//{
		totalDistance<<<1,cities>>>(dev_solutions,dev_distances,dev_x,dev_y);
		evaluateFitness<<<1,1>>>(dev_distances,dev_fitness);
		cudaMemcpy(fitness,dev_fitness,chromosomes*sizeof(double),cudaMemcpyDeviceToHost);
		for(i=0;i<chromosomes;i++)
			sum+=*(fitness+i);
		maxFitPos=(maxelement(fitness,fitness+chromosomes) - fitness);
		newFit=*(fitness+maxFitPos);
		curandGenerateUniformDouble(gen, dev_rf1, chromosomes);
		curandGenerateUniformDouble(gen, dev_rf2, chromosomes);
		curandGenerateUniformDouble(gen, dev_r1, chromosomes);
		curandGenerateUniformDouble(gen, dev_rc, chromosomes);
		nextPopulation<<<1,1>>>(dev_solutions,dev_newPopulation,dev_bestchromosome,dev_fitness,crossoverProbability,mutationProbability,dev_rf1,dev_rf2,dev_r1,dev_rc,sum);
		generations++;
		/*if(newFit > bestFitness)
		{
			bestFitness = newFit;
			generationsWithoutImprovement = 0;
			printf("Best fitness:%lf\n",bestFitness);
		}
		else
		{
			++generationsWithoutImprovement;
		}
	}*/
	cudaMemcpy(fitness,dev_fitness,chromosomes*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(bestchromosome,dev_bestchromosome,cities*sizeof(int),cudaMemcpyDeviceToHost);
	maxFitPos=(maxelement(fitness,fitness+chromosomes) - fitness);
	newFit=*(fitness+maxFitPos);
	printf("DONE\n");
	printf("Number of generations: %d\n",generations);
	printf("Best chromosome info: \n");
	printf("\t-Path: ");
	for(i=0;i<cities;i++)
		printf(" %d ",*(bestchromosome+i));
	printf("\n");
	printf("\t-Goal function: %lf\n",newFit);
	printchrom<<<1,1>>>(dev_solutions);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_fitness);
	cudaFree(dev_solutions);
	cudaFree(dev_distances);
	cudaFree(dev_rf1);
	cudaFree(dev_rf2);
	cudaFree(dev_r1);
	cudaFree(dev_rc);
	cudaFree(dev_newPopulation);
	cudaFree(dev_bestchromosome);
	free(citiesX);
	free(citiesY);
	free(fitness);
	free(bestchromosome);
}
