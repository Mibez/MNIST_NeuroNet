// +++++++++++++++++++++++++++++++++++++++++++++++++++++
// Title: FileReader.c
// Author: Miikka Lukumies
// Description: Custom file loader for MNIST database files
// +++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <stdio.h>
#include <stdlib.h>
#include "FileReader.h"

double doublify(unsigned char num, int coeff, double norm);

int readFile(char *path, int width, int headerSize, int amount, double **destination, int coefficient, double normalization)
{
	int errorMessage = 0;
	FILE *file;
	unsigned char **buffer;
	unsigned char *header;
	file = fopen(path, "rb");

	// allocate memory
	buffer = (unsigned char **) calloc(amount , sizeof(unsigned char *));
	for(int i = 0; i < amount; i++)
	{
		buffer[i] = (unsigned char*)calloc(width, sizeof(unsigned char));

	}

	header = (unsigned char *)calloc(headerSize, sizeof(unsigned char));
	if((!buffer) || (!header)) errorMessage = -1;

	if(errorMessage == 0)
	{

		fread(header, 1, headerSize, file); // skip the header

		for(int i = 0; i < amount; i++)
		{
			fread(buffer[i], 1, width, file);
		}

		for(int i = 0; i < amount; i++)
		{
			double *doubleValues = (double*) calloc(width+1, sizeof(double));

			// make the read buffer one-dimensional, make it double and normalize it
			// as well as leave the first index empty for index compatibility with
			// the rest of the code
			for(int j = 0; j < width; j++)
			{
				doubleValues[j+1] = doublify(buffer[i][j], coefficient, normalization);
			}

			destination[i] = doubleValues;
		}

	}


	// clean-up
	for(int i = 0; i < amount; i++)
	{
		free(buffer[i]);
	}
	free(buffer);
	free(header);
	return errorMessage;
}

double doublify(unsigned char num, int coeff, double norm)
{
	return  ((1.0 / (double)coeff) * (int)num - norm);
}
