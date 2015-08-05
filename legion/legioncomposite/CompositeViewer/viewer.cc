/**
 * Ian Sohl - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Image Viewing Code
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <Magick++.h>
#include <sstream>

using namespace std;
using namespace Magick;


#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
		( std::ostringstream() << std::dec << x ) ).str()


float charTofloat(char b0, char b1, char b2, char b3){
	union{
		float d;
		char b[4];
	} u;
	u.b[0] = b0;
	u.b[1] = b1;
	u.b[2] = b2;
	u.b[3] = b3;
	return u.d;
}

int charToInt(char a0, char a1, char a2, char a3){
	union{
		int i;
		char a[4];
	} u;
	u.a[0] = a0;
	u.a[1] = a1;
	u.a[2] = a2;
	u.a[3] = a3;
	return u.i;
}

int main(int argc, char **argv){
	InitializeMagick(*argv);
	ifstream inFile("../output.raw", ios::in | ios::binary | ios::ate);
	if (inFile.is_open()){
		streampos size = inFile.tellg();
		cout << "Reading in " << size << " bytes" << endl;
		char * pdata = new char[size];
		inFile.seekg(0, ios::beg);
		inFile.read(&pdata[0], size);
		inFile.close();
		int width = charToInt(pdata[0],pdata[1],pdata[2],pdata[3]);
		int height = charToInt(pdata[4],pdata[5],pdata[6],pdata[7]);
		cout << "\tData read successful" << endl;
		Image image(SSTR(width << "x" << height), "white");
		image.matte(true);
		int offset = 8;
		int x = 0, y = 0;
		for(int i = 0; i < width*height*4;){
			int rpos = (i++)*sizeof(float) + offset;
			int gpos = (i++)*sizeof(float) + offset;
			int bpos = (i++)*sizeof(float) + offset;
			int apos = (i++)*sizeof(float) + offset;
			float b = charTofloat(pdata[bpos+0],pdata[bpos+1],pdata[bpos+2],pdata[bpos+3]);
			float g = charTofloat(pdata[gpos+0],pdata[gpos+1],pdata[gpos+2],pdata[gpos+3]);
			float r = charTofloat(pdata[rpos+0],pdata[rpos+1],pdata[rpos+2],pdata[rpos+3]);
			float a = 1.0 - charTofloat(pdata[apos+0],pdata[apos+1],pdata[apos+2],pdata[apos+3]);
			short rc = static_cast<short>(r*65535);
			short gc = static_cast<short>(g*65535);
			short bc = static_cast<short>(b*65535);
			short ac = static_cast<short>(a*65535);
			image.pixelColor(x,y,Color(rc,gc,bc,ac));
			++x;
			if(x==width){
				x = 0;
				++y;
			}
		}
		image.write("output.png");
		cout << "\tImage written to file" << endl;
		return 1;
	}
	cout << "Failed to load RAW file" << endl;
	return 0;
}
