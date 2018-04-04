// +++++++++++++++++++++++++++++++++++++++++++++++++
// Title: FileReader.h
// Author: Miikka Lukumies
// Description: Header file for FileReader.c
// +++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef FILEREADER_H
#define FILEREADER_H

int readFile(char *path, int width, int headerSize, int amount, double **destination, int coefficient, double normalization);

#endif
