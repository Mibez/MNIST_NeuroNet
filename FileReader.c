#include <stdio.h>
#include <stdlib.h>
#include "FileReader.h"

double doublify(unsigned char num, int coeff, double norm);

int readFile(char *path, int width, int headerSize, int amount, double **destination, int coefficient, double normalization)
{
	int errorMessage = 0;
	FILE *file;
	unsigned char **buffer;
	unsigned char *temp;
	file = fopen(path, "rb");

	buffer = (unsigned char **) calloc(amount , sizeof(unsigned char *));
	for(int i = 0; i < amount; i++)
	{
		buffer[i] = (unsigned char*)calloc(width, sizeof(unsigned char));

	}

	temp = (unsigned char *)calloc(headerSize, sizeof(unsigned char));
	if((!buffer) || (!temp)) errorMessage = -1;

	if(errorMessage == 0)
	{
		printf("\nReading file...");

		fread(temp, 1, headerSize, file); // skip the header

		for(int i = 0; i < amount; i++)
		{
			fread(buffer[i], 1, width, file);
		}
		printf("Done!\n");

		for(int i = 0; i < amount; i++)
		{
			double *tempD = (double*) calloc(width+1, sizeof(double));

			for(int j = 0; j < width; j++)
			{
				tempD[j+1] = doublify(buffer[i][j], coefficient, normalization);
			}

			destination[i] = tempD;
		}

	}


	for(int i = 0; i < amount; i++)
	{
		free(buffer[i]);
	}
	free(buffer);
	free(temp);
	return errorMessage;
}

double doublify(unsigned char num, int coeff, double norm)
{
	return  ((1.0 / (double)coeff) * (int)num - norm);
}
