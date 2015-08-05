/**
 * Ian Sohl - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Main Code
 */


#include "composite.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sys/time.h>
#include <algorithm>
#include <unistd.h>
#include "DataMgr.h"


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace std;


void create_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime); // Override of the GPU rendering task

void create_isosurface_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime); // Override of the GPU rendering task

void render_isosurface_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime); // Override of the GPU rendering task

int sqr(int x){
	/**
	 * Returns an integer square value
	 */
	return x*x;
}


int subCoordTransform(int z, int width, int yoffset){
	/**
	 * Converts a index in total region space to an index in sub-region space
	 */
	return z - yoffset * width;
}



float exclusion(float a){
	/**
	 * Compositing swap function
	 * Computes 1 - value
	 */
	return 1.0-a;
}

float passthrough(float a){
	/**
	 * Compositing swap function
	 * Passes through same value
	 */
	return a;
}

float pass1(float a){
	/**
	 * Compositing swap function
	 * Returns 1.0;
	 */
	return 1.0;
}

float pass0(float a){
	/**
	 * Compositing swap function
	 * Returns 0.0;
	 */
	return 0.0;
}

void create_interface_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime){
	compositeArguments co = *((compositeArguments*)task->args);	// Task metadata
#ifndef ISOSURFACE
	TaskLauncher loadLauncher(CREATE_TASK_ID, TaskArgument(&co,sizeof(co)));	// Spawn the renderer task
	loadLauncher.add_region_requirement(RegionRequirement(regions[0].get_logical_region(),READ_ONLY,EXCLUSIVE,regions[0].get_logical_region()));
	loadLauncher.add_field(0,FID_META);		// Metadata as first region
	loadLauncher.add_region_requirement(RegionRequirement(regions[1].get_logical_region(),WRITE_DISCARD,EXCLUSIVE,regions[1].get_logical_region()));
	loadLauncher.add_field(1,FID_VAL);		// Output Image as second region
	loadLauncher.add_region_requirement(RegionRequirement(regions[2].get_logical_region(),READ_ONLY,EXCLUSIVE,regions[2].get_logical_region()));
	loadLauncher.add_field(2,FID_VAL);		// Input Data as third region
	runtime->execute_task(ctx,loadLauncher);	// Launch and terminate compositor task
#else
	PhysicalRegion metadataPhysicalRegion = regions[0];
	LogicalRegion metadataLogicalRegion = metadataPhysicalRegion.get_logical_region();
	IndexSpace metadataIndexSpace = metadataLogicalRegion.get_index_space();
	Domain totalDomain = runtime->get_index_space_domain(ctx,metadataIndexSpace);
	Rect<1> totalRect = totalDomain.get_rect<1>();	// Get metadata value index
	RegionAccessor<AccessorType::Generic, Image> filenameAccessor = regions[0].get_field_accessor(FID_META).typeify<Image>();
	Image tmpimg = filenameAccessor.read(DomainPoint::from_point<1>(Point<1>(totalRect.lo.x[0])));	// Metadata for current render

	Rect<1> dataBound = Rect<1>(0,(tmpimg.partition.xmax-tmpimg.partition.xmin)*(tmpimg.partition.ymax-tmpimg.partition.ymin)*(tmpimg.partition.zmax-tmpimg.partition.zmin)-1);
	IndexSpace dataIndexSpace = runtime->create_index_space(ctx, Domain::from_rect<1>(dataBound)); //Create the Index Space (1 index per voxel)
	FieldSpace dataFieldSpace = runtime->create_field_space(ctx);	// Simple field space
	{
		FieldAllocator allocator = runtime->create_field_allocator(ctx,dataFieldSpace);
		allocator.allocate_field(3*sizeof(float),FID_VERTEX);
		allocator.allocate_field(3*sizeof(float),FID_NORMAL);
	}
	LogicalRegion dataLogicalRegion = runtime->create_logical_region(ctx,dataIndexSpace,dataFieldSpace); // Create the Logical Region


	Rect<1> ntriDataBound = Rect<1>(0,0);
	IndexSpace ntriDataIndexSpace = runtime->create_index_space(ctx, Domain::from_rect<1>(ntriDataBound)); //Create the Index Space (1 index per voxel)
	FieldSpace ntriDataFieldSpace = runtime->create_field_space(ctx);	// Simple field space
	{
		FieldAllocator allocator = runtime->create_field_allocator(ctx,ntriDataFieldSpace);
		allocator.allocate_field(sizeof(int),FID_NTRI);
	}
	LogicalRegion ntriDataLogicalRegion = runtime->create_logical_region(ctx,ntriDataIndexSpace, ntriDataFieldSpace); // Create the Logical Region


	TaskLauncher loadLauncher(CREATE_ISOSURFACE_TASK_ID, TaskArgument(&co,sizeof(co)));	// Spawn the renderer task
	loadLauncher.add_region_requirement(RegionRequirement(metadataLogicalRegion,READ_ONLY,EXCLUSIVE,metadataLogicalRegion));
	loadLauncher.add_field(0,FID_META);		// Metadata as first region
	loadLauncher.add_region_requirement(RegionRequirement(dataLogicalRegion,WRITE_DISCARD,EXCLUSIVE,dataLogicalRegion));
	loadLauncher.add_field(1,FID_VERTEX);		// Output Isosurface data
	loadLauncher.add_field(1,FID_NORMAL);
	loadLauncher.add_region_requirement(RegionRequirement(regions[2].get_logical_region(),READ_ONLY,EXCLUSIVE,regions[2].get_logical_region()));
	loadLauncher.add_field(2,FID_VAL);		// Input Data as third region
	loadLauncher.add_region_requirement(RegionRequirement(ntriDataLogicalRegion,WRITE_DISCARD,EXCLUSIVE,ntriDataLogicalRegion));
	loadLauncher.add_field(3,FID_NTRI);		// Output Isosurface data
	runtime->execute_task(ctx,loadLauncher);	// Launch and terminate compositor task


	TaskLauncher renderLauncher(RENDER_ISOSURFACE_TASK_ID, TaskArgument(&co,sizeof(co)));	// Spawn the renderer task
	renderLauncher.add_region_requirement(RegionRequirement(metadataLogicalRegion,READ_ONLY,EXCLUSIVE,metadataLogicalRegion));
	renderLauncher.add_field(0,FID_META);		// Metadata as first region
	renderLauncher.add_region_requirement(RegionRequirement(regions[1].get_logical_region(),WRITE_DISCARD,EXCLUSIVE,regions[1].get_logical_region()));
	renderLauncher.add_field(1,FID_VAL);		// Output Image as second region
	renderLauncher.add_region_requirement(RegionRequirement(dataLogicalRegion,READ_ONLY,EXCLUSIVE,dataLogicalRegion));
	renderLauncher.add_field(2,FID_VERTEX);		// Input Isosurface data
	renderLauncher.add_field(2,FID_NORMAL);
	renderLauncher.add_region_requirement(RegionRequirement(ntriDataLogicalRegion,READ_ONLY,EXCLUSIVE,ntriDataLogicalRegion));
	renderLauncher.add_field(3,FID_NTRI);		// Output Isosurface data
	runtime->execute_task(ctx,renderLauncher);	// Launch and terminate compositor task

#endif

}

void composite(RegionAccessor<AccessorType::Generic, float> input1, RegionAccessor<AccessorType::Generic, float> input2, RegionAccessor<AccessorType::Generic, float> O, int start, int stop, tripleArgument ta, float (*FA)(float), float (*FB)(float)){
	/**
	 * Generic alpha compositing that can be called with adjustable functions for different operations
	 */
	int linewidth = 4 * ta.co.width;	// Number of indices on one scanline

	int voffset1 = (ta.co1.miny - ta.co.miny);	// Vertical offset of region 1 from the top
	int maxbound1 = linewidth * (ta.co1.maxy - ta.co1.miny+1);	// Vertical size of region 1

	int voffset2 = (ta.co2.miny - ta.co.miny); 	// Vertical offset of region 2 from the top
	int maxbound2 = linewidth * (ta.co2.maxy - ta.co2.miny+1);	// Vertical size of region 2


	Point<1> input1Point(subCoordTransform(start,linewidth,voffset1));	// Scanning current point in region 1
	Point<1> input2Point(subCoordTransform(start,linewidth,voffset2));	// Scanning point for region 2
	Point<1> OPoint(start); 											// Scanning point for output region


	for(int i = start; i < stop;i+=4){	//  Step through all of the data in groups of 4 (RGBA pixels)

		bool range1 = input1Point.x[0] >= 0 && input1Point.x[0] < maxbound1; // Check if region 1 is in range
		bool range2 = input2Point.x[0] >= 0 && input2Point.x[0] < maxbound2; // Check if region 2 is in range

#ifdef ISOSURFACE
		bool transparent1 = true;
		bool transparent2 = true;
		for(int j = 0; j < 3; ++j){
			if(range1 ? input1.read(DomainPoint::from_point<1>(input1Point)) : 0.0!=0.0){
				transparent1 = false;
			}
			if(range2 ? input2.read(DomainPoint::from_point<1>(input2Point)) : 0.0!=0.0){
				transparent2 = false;
			}
			input1Point.x[0]++;
			input2Point.x[0]++;
		}
		float alphaA = 0.0;
		float alphaB = 0.0;
		if(!transparent1){
			alphaA = range1 ? input1.read(DomainPoint::from_point<1>(input1Point)) : 0.0;
		}
		if(!transparent2){
			alphaB = range2 ? input2.read(DomainPoint::from_point<1>(input2Point)) : 0.0;
		}
#else
		input1Point.x[0]+=3;	// Increment by 3 to get the alpha value
		input2Point.x[0]+=3;
		float alphaA = range1 ? input1.read(DomainPoint::from_point<1>(input1Point)) : 0.0; // If in range, get the alpha value
		float alphaB = range2 ? input2.read(DomainPoint::from_point<1>(input2Point)) : 0.0; //		Otherwise alpha is 0.0
#endif
		input1Point.x[0]-=3;	// Step back in place for the red value
		input2Point.x[0]-=3;
		float alphaC = alphaA*FA(alphaB)+alphaB*FB(alphaA); // Compute the output alpha
		if(alphaC!=0){			// If there is a non-zero alpha
			for(int j = 0; j < 3; ++j){	// For each of R, G, B
				float A = range1 ? input1.read(DomainPoint::from_point<1>(input1Point)) : 0.0; // Get the values from the input
				float B = range2 ? input2.read(DomainPoint::from_point<1>(input2Point)) : 0.0;
				O.write(DomainPoint::from_point<1>(OPoint),(A*alphaA*FA(alphaB)+B*alphaB*FB(alphaA))/alphaC); // Compute composite and write
				input1Point.x[0]++; // Step to next point
				input2Point.x[0]++;
				OPoint.x[0]++;
			}
		}
		else{	// If Alpha is zero
			for(int j = 0; j < 3; ++j){
				O.write(DomainPoint::from_point<1>(OPoint),0.0); // Fill RGB with zeros
				OPoint.x[0]++;
			}
			input1Point.x[0]+=3;
			input2Point.x[0]+=3;
		}
		O.write(DomainPoint::from_point<1>(OPoint),alphaC); // Write output alpha
		OPoint.x[0]++;		// Increment for next pixel
		input1Point.x[0]++;
		input2Point.x[0]++;
	}
}

void compositeOver(RegionAccessor<AccessorType::Generic, float> input1, RegionAccessor<AccessorType::Generic, float> input2, RegionAccessor<AccessorType::Generic, float> imgO, int start, int stop, tripleArgument ta){
	/**
	 *  Alpha 'Over' Compositing
	 */
	composite(input1, input2,imgO,start,stop,ta,&pass1,&exclusion);
}

void combine_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime){
	/**
	 * Combining task that actually composites images together
	 */
	assert(regions.size()==3);
	tripleArgument ta = *((tripleArgument*)task->args); // Get metadata properties
	Domain outDomain = runtime->get_index_space_domain(ctx,regions[2].get_logical_region().get_index_space());
	Rect<1> outRect = outDomain.get_rect<1>();			// Get the size of the return image
	RegionAccessor<AccessorType::Generic,float> inputAccessor1 = regions[0].get_field_accessor(FID_VAL).typeify<float>();
	RegionAccessor<AccessorType::Generic,float> inputAccessor2 = regions[1].get_field_accessor(FID_VAL).typeify<float>();
	RegionAccessor<AccessorType::Generic,float> outputAccessor = regions[2].get_field_accessor(FID_VAL).typeify<float>();
	compositeOver(inputAccessor1,inputAccessor2,outputAccessor,outRect.lo.x[0],outRect.hi.x[0],ta); // Call the Composite 'Over' version
}




void composite_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime){
	/**
	 * Main recursive 'compositing' task.
	 * This is an inner task with control structures only that don't access data.
	 */
	compositeArguments co = *((compositeArguments*)task->args);	// Task metadata
	int size = (co.width)*(co.maxy-co.miny+1)*4;				// Total image pixel count
	int inputRegionCount = regions.size();
	PhysicalRegion metadataPhysicalRegion = regions[0];
	LogicalRegion metadataLogicalRegion = metadataPhysicalRegion.get_logical_region();
	IndexSpace metadataIndexSpace = metadataLogicalRegion.get_index_space();
	Domain totalDomain = runtime->get_index_space_domain(ctx,metadataIndexSpace);
	Rect<1> totalRect = totalDomain.get_rect<1>();				// Get the size of the metadata region

	if(totalRect.lo.x[0]==totalRect.hi.x[0]){					// If the metadata region only has 1 element in it
		int i = totalRect.lo.x[0]+2;
		IndexSpace inputIndex = regions[i].get_logical_region().get_index_space();
		Domain inputDomain = runtime->get_index_space_domain(ctx,inputIndex);
		Rect<1> inputRect = inputDomain.get_rect<1>();
		RegionAccessor<AccessorType::Generic,float> inputAccessor = regions[i].get_field_accessor(FID_VAL).typeify<float>();
		RegionAccessor<AccessorType::Generic,float> imgAccessor = regions[1].get_field_accessor(FID_VAL).typeify<float>();
//		RegionAccessor<AccessorType::Generic,Image> metadataAccessor = regions[0].get_field_accessor(FID_META).typeify<Image>();
//		cout << "Compositing: " << metadataAccessor.read(DomainPoint::from_point<1>(totalRect.lo.x[0])).order << endl;
		for(GenericPointInRectIterator<1>pir(inputRect); pir; pir++){
			imgAccessor.write(DomainPoint::from_point<1>(pir.p),inputAccessor.read(DomainPoint::from_point<1>(pir.p)));
		}
	}
	else{
		FieldSpace metadataOutputField = runtime->create_field_space(ctx); // Setup field spaces for the metadata values
		{
			FieldAllocator allocator = runtime->create_field_allocator(ctx,metadataOutputField);
			allocator.allocate_field(sizeof(Image),FID_META);
		}
		// Create index spaces for splitting the metadata into two (roughly equivalent) halves.
		IndexSpace metadataOutputIndex1 = runtime->create_index_space(ctx, Domain::from_rect<1>(Rect<1>(Point<1>(totalRect.lo.x[0]),Point<1>(static_cast<int>((totalRect.hi.x[0]-totalRect.lo.x[0])/2)+totalRect.lo.x[0]))));
		LogicalRegion metadataOutputLogicalRegion1 = runtime->create_logical_region(ctx,metadataOutputIndex1,metadataOutputField);
		// And bind them to new logical regions
		IndexSpace metadataOutputIndex2 = runtime->create_index_space(ctx, Domain::from_rect<1>(Rect<1>(Point<1>(static_cast<int>((totalRect.hi.x[0]-totalRect.lo.x[0])/2)+totalRect.lo.x[0]+1),Point<1>(totalRect.hi.x[0]))));
		LogicalRegion metadataOutputLogicalRegion2 = runtime->create_logical_region(ctx,metadataOutputIndex2,metadataOutputField);

		compositeArguments co1 = co;	// Create sub-arguments for the two new tasks
		compositeArguments co2 = co;

		// Map and prepare to copy the metadata values
		RegionAccessor<AccessorType::Generic, Image> filenameAccessor = regions[0].get_field_accessor(FID_META).typeify<Image>();

		for(int r = 0; r < 2; ++r){		// Setup for the two new tasks
			RegionRequirement req;
			Domain metadataDomain;
			if(r==0){					// Bind into the first logical region
				req = RegionRequirement(metadataOutputLogicalRegion1,WRITE_DISCARD,EXCLUSIVE,metadataOutputLogicalRegion1);
				metadataDomain = runtime->get_index_space_domain(ctx,metadataOutputIndex1);
			}
			else{						// Or the second region
				req = RegionRequirement(metadataOutputLogicalRegion2,WRITE_DISCARD,EXCLUSIVE,metadataOutputLogicalRegion2);
				metadataDomain = runtime->get_index_space_domain(ctx,metadataOutputIndex2);
			}
			req.add_field(FID_META);	// Need to copy metadata values
			InlineLauncher metadataLauncher(req);
			PhysicalRegion metadataPhysicalRegion = runtime->map_region(ctx,metadataLauncher);
			metadataPhysicalRegion.wait_until_valid();
			RegionAccessor<AccessorType::Generic, Image> accessFilename = metadataPhysicalRegion.get_field_accessor(FID_META).typeify<Image>();
			Rect<1> metadataBound = metadataDomain.get_rect<1>();	// The range of indices of the metadata region

			/// Update the relative image bounds based on data bounds (does nothing for now)
			int miny = co.height;	// Set to opposite extrema
			int maxy = 0;
			for(GenericPointInRectIterator<1> pir(metadataBound); pir; pir++){ // Iterate through images and expand bounds as necessary
				Image tmpimg = filenameAccessor.read(DomainPoint::from_point<1>(pir.p));
				miny = min(tmpimg.ymin,miny);
				maxy = max(tmpimg.ymax,maxy);
				accessFilename.write(DomainPoint::from_point<1>(pir.p),tmpimg);
			}
			if(r==0){	// Define the corresponding sub-arguments
				co1.miny = miny;
				co1.maxy = maxy;
			}
			else{
				co2.miny = miny;
				co2.maxy = maxy;
			}
			runtime->unmap_region(ctx,metadataPhysicalRegion);	// Free up resources
		}


		FieldSpace inputField = runtime->create_field_space(ctx);	// Prepare the image field spaces
		{
			FieldAllocator allocator = runtime->create_field_allocator(ctx,inputField);
			allocator.allocate_field(sizeof(float),FID_VAL);
		}

		Rect<1> inputBound1(Point<1>(0),Point<1>((co.width)*(co1.maxy-co1.miny+1)*4-1)); // Create a new region for the return images
		IndexSpace inputIndex1 = runtime->create_index_space(ctx, Domain::from_rect<1>(inputBound1));

		Rect<1> inputBound2(Point<1>(0),Point<1>((co.width)*(co2.maxy-co2.miny+1)*4-1)); // One for each sub-task
		IndexSpace inputIndex2 = runtime->create_index_space(ctx, Domain::from_rect<1>(inputBound2));

		LogicalRegion imgLogicalRegion1 = runtime->create_logical_region(ctx,inputIndex1,inputField);
		LogicalRegion imgLogicalRegion2 = runtime->create_logical_region(ctx,inputIndex2,inputField);




		TaskLauncher compositeLauncher1(COMPOSITE_TASK_ID, TaskArgument(&co1,sizeof(co1))); // Launch a single task for each half of the remaining data
		compositeLauncher1.add_region_requirement(RegionRequirement(metadataOutputLogicalRegion1,READ_ONLY,EXCLUSIVE,metadataOutputLogicalRegion1));
		compositeLauncher1.add_field(0,FID_META); // Recursively calling composite task, so same arguments
		compositeLauncher1.add_region_requirement(RegionRequirement(imgLogicalRegion1,WRITE_DISCARD,EXCLUSIVE,imgLogicalRegion1));
		compositeLauncher1.add_field(1,FID_VAL);
		for(int i = 2; i < inputRegionCount; ++i){
			compositeLauncher1.add_region_requirement(RegionRequirement(regions[i].get_logical_region(),READ_ONLY,EXCLUSIVE,regions[i].get_logical_region()));
			compositeLauncher1.add_field(i,FID_VAL);
		}
		runtime->execute_task(ctx,compositeLauncher1);

		TaskLauncher compositeLauncher2(COMPOSITE_TASK_ID, TaskArgument(&co2,sizeof(co2))); // Second half of metadata
		compositeLauncher2.add_region_requirement(RegionRequirement(metadataOutputLogicalRegion2,READ_ONLY,EXCLUSIVE,metadataOutputLogicalRegion2));
		compositeLauncher2.add_field(0,FID_META);
		compositeLauncher2.add_region_requirement(RegionRequirement(imgLogicalRegion2,WRITE_DISCARD,EXCLUSIVE,imgLogicalRegion2));
		compositeLauncher2.add_field(1,FID_VAL);
		for(int i = 2; i < inputRegionCount; ++i){
			compositeLauncher2.add_region_requirement(RegionRequirement(regions[i].get_logical_region(),READ_ONLY,EXCLUSIVE,regions[i].get_logical_region()));
			compositeLauncher2.add_field(i,FID_VAL);
		}
		runtime->execute_task(ctx,compositeLauncher2);


		/// Actual Compositing

		tripleArgument ta = {co,co1,co2}; // New argument type that combines all possible metadata for the compositing operation

		LogicalRegion combineOutputRegion = regions[1].get_logical_region(); // Image return region

		const int divisions = 8; // This is an arbitrarily chosen number based on Darwin configuration (Nodes with 32 cores)

		Rect<1> combineColorRect(Point<1>(0),Point<1>(divisions-1)); // Define the number of coloring divisions
		Domain combineColorDomain = Domain::from_rect<1>(combineColorRect);

		DomainColoring combineInputColoring1; // Need three separate colorings for each of the regions
		DomainColoring combineInputColoring2; // 	Since the regions don't necessarily completely overlap
		DomainColoring combineOutputColoring;
		{
			int index = 0;	// Coloring current pixel value
			int linewidth = 4 * co.width; // Size of an image scan line
			int partitionCount = (int)(((int)(size/4)/divisions)*4/linewidth)*linewidth; // Number of pixels per partition

			int voffset1 = (co1.miny - co.miny); // Vertical offset of the first region
			int maxbound1 = linewidth * (co1.maxy - co1.miny+1); // Vertical size of the first region

			int voffset2 = (co2.miny - co.miny); // Vertical offset of the second region
			int maxbound2 = linewidth * (co2.maxy - co2.miny+1); // Vertical size of the second region

			for(int i = 0; i < divisions-1; ++i){ // Step through each color
				Rect<1> subrect(Point<1>(index),Point<1>(index+partitionCount-1)); 	// Find the size of the output region in coloring
				combineOutputColoring[i] = Domain::from_rect<1>(subrect); 			// Get a domain of this area
				int lower1 = subCoordTransform(index,linewidth,voffset1);			// Get the lower bound index for the first input region
				int upper1 = subCoordTransform(index+partitionCount-1,linewidth,voffset1); // And the upper bound index
				if(upper1 < 0 || lower1 >= maxbound1){								// Check if the bounds fit
					Rect<1> subrect1(Point<1>(0),Point<1>(0));						// If not, the coloring doesn't overlap the region
					combineInputColoring1[i] = Domain::from_rect<1>(subrect1);		// Define a hack to indicate that to the composite task
				}
				else{																// Otherwise define the mapping
					lower1 = max(lower1,0);											// Clamp it to the bounds of the region
					upper1 = min(upper1,maxbound1-1);
					Rect<1> subrect1(Point<1>(lower1),Point<1>(upper1-0));			// Find the actual rectangle of the first region
					combineInputColoring1[i] = Domain::from_rect<1>(subrect1);		// And define it in the coloring domain
				}
				int lower2 = subCoordTransform(index,linewidth,voffset2);			// Complete the same process for the second input region
				int upper2 = subCoordTransform(index+partitionCount-1,linewidth,voffset2);
				if(upper2 < 0 || lower2 >= maxbound2){
					Rect<1> subrect2(Point<1>(0),Point<1>(0));
					combineInputColoring2[i] = Domain::from_rect<1>(subrect2);
				}
				else{
					lower2 = max(lower2,0);
					upper2 = min(upper2,maxbound2-1);
					Rect<1> subrect2(Point<1>(lower2),Point<1>(upper2-0));
					combineInputColoring2[i] = Domain::from_rect<1>(subrect2);
				}
				index += partitionCount;											// Keep track of the current index
			}
			Rect<1> subrect(Point<1>(index),Point<1>(size-1));						// In case we have any extra space, fit it all into the last color
			combineOutputColoring[divisions-1] = Domain::from_rect<1>(subrect);

			int lower1 = subCoordTransform(index,linewidth,voffset1);				// Same process, just with everything until the end
			int upper1 = subCoordTransform(size-1,linewidth,voffset1);
			if(upper1 < 0 || lower1 >= maxbound1){
				Rect<1> subrect1(Point<1>(0),Point<1>(0));
				combineInputColoring1[divisions-1] = Domain::from_rect<1>(subrect1);
			}
			else{
				lower1 = max(lower1,0);
				upper1 = min(upper1,maxbound1-1);
				Rect<1> subrect1(Point<1>(lower1),Point<1>(upper1-0));
				combineInputColoring1[divisions-1] = Domain::from_rect<1>(subrect1);
			}
			int lower2 = subCoordTransform(index,linewidth,voffset2);
			int upper2 = subCoordTransform(size-1,linewidth,voffset2);
			if(upper2 < 0 || lower2 >= maxbound2){
				Rect<1> subrect2(Point<1>(0),Point<1>(0));
				combineInputColoring2[divisions-1] = Domain::from_rect<1>(subrect2);
			}
			else{
				lower2 = max(lower2,0);
				upper2 = min(upper2,maxbound2-1);
				Rect<1> subrect2(Point<1>(lower2),Point<1>(upper2+0));
				combineInputColoring2[divisions-1] = Domain::from_rect<1>(subrect2);
			}
		}

		/// Pass the individual colorings into Index Partitions
		IndexPartition combineInputIndexPartition1 = runtime->create_index_partition(ctx, inputIndex1, combineColorDomain, combineInputColoring1, false);
		IndexPartition combineInputIndexPartition2 = runtime->create_index_partition(ctx, inputIndex2, combineColorDomain, combineInputColoring2, false);
		IndexPartition combineOutputIndexPartition = runtime->create_index_partition(ctx, combineOutputRegion.get_index_space(), combineColorDomain, combineOutputColoring, true);

		/// And define logical partitions on those index partitions
		LogicalPartition combineInputLogicalPartition1 = runtime->get_logical_partition(ctx, imgLogicalRegion1, combineInputIndexPartition1);
		LogicalPartition combineInputLogicalPartition2 = runtime->get_logical_partition(ctx, imgLogicalRegion2, combineInputIndexPartition2);
		LogicalPartition combineOutputLogicalPartition = runtime->get_logical_partition(ctx, combineOutputRegion, combineOutputIndexPartition);


		ArgumentMap argMap;

		/// Index launcher for the actual combinator tasks
		IndexLauncher combineLauncher(COMBINE_TASK_ID, combineColorDomain,TaskArgument(&ta,sizeof(ta)), argMap);
		combineLauncher.add_region_requirement(RegionRequirement(combineInputLogicalPartition1,0,READ_ONLY,EXCLUSIVE,imgLogicalRegion1));
		combineLauncher.add_field(0,FID_VAL);	// First region is the first input region
		combineLauncher.add_region_requirement(RegionRequirement(combineInputLogicalPartition2,0,READ_ONLY,EXCLUSIVE,imgLogicalRegion2));
		combineLauncher.add_field(1,FID_VAL);	// Second region is the other input region
		combineLauncher.add_region_requirement(RegionRequirement(combineOutputLogicalPartition,0,WRITE_DISCARD,EXCLUSIVE,combineOutputRegion));
		combineLauncher.add_field(2,FID_VAL);	// Third region is the output image region
		runtime->execute_index_space(ctx,combineLauncher);

		/// Clean up created components afterwards
		runtime->destroy_logical_region(ctx,metadataOutputLogicalRegion2);
		runtime->destroy_logical_region(ctx,metadataOutputLogicalRegion1);
		runtime->destroy_field_space(ctx,metadataOutputField);
		runtime->destroy_index_space(ctx,metadataOutputIndex2);
		runtime->destroy_index_space(ctx,metadataOutputIndex1);
		runtime->destroy_logical_region(ctx,imgLogicalRegion2);
		runtime->destroy_logical_region(ctx,imgLogicalRegion1);
		runtime->destroy_index_space(ctx,inputIndex1);
		runtime->destroy_index_space(ctx,inputIndex2);
		runtime->destroy_field_space(ctx,inputField);

	}
}



void display_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime){
	/**
	 * Task for sending data to Qt Window
	 */
	compositeArguments co = *((compositeArguments*)task->args); // Load the image metadata
	PhysicalRegion imgPhysicalRegion = regions[0];				// The region that holds the image pixels
	LogicalRegion imgLogicalRegion = imgPhysicalRegion.get_logical_region();
	IndexSpace imgIndexSpace = imgLogicalRegion.get_index_space();
	Domain imgDomain = runtime->get_index_space_domain(ctx,imgIndexSpace);
	Rect<1> imgBound = imgDomain.get_rect<1>();					// Get the size of the pixel data
#ifdef QT
	RegionAccessor<AccessorType::Generic,float> accessImg = imgPhysicalRegion.get_field_accessor(FID_VAL).typeify<float>();
	int *vals = new int[imgBound.hi.x[0]];						// Define an array for passing data
	int i = 0;
	for(GenericPointInRectIterator<1> pir(imgBound); pir; pir++){
		int val = (int)(accessImg.read(DomainPoint::from_point<1>(pir.p))*255); // Bind into the array (and convert to integer) [0-255]
		vals[i++] = val;
	}
	cout << "Sending" << endl;
	newImage(vals,co.mov,co.width,co.height);					// Use the interface to pass the reference to Qt
#else
	cout << "Writing to File" << endl;
	char filename[50];
	sprintf(filename,"output.raw");
	ofstream oFile(filename, ios::out | ios::binary);
	oFile.write(reinterpret_cast<char*>(&co.width),sizeof(int));
	oFile.write(reinterpret_cast<char*>(&co.height),sizeof(int));
	for(GenericPointInRectIterator<1> pir(imgBound); pir; pir++){
		float val = accessImg.read(DomainPoint::from_point<1>(pir.p));
		oFile.write(reinterpret_cast<char*>(&val),sizeof(float));
	}
	oFile.close();
#endif
}



void coherence_simulation_task(const Task*task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime){
	compositeArguments co = *((compositeArguments*)task->args);

	LogicalRegion outputLogicalRegion = task->regions[1].region;

	AcquireLauncher acqLauncher(outputLogicalRegion,outputLogicalRegion,regions[1]);
	acqLauncher.add_field(FID_VAL);
	runtime->issue_acquire(ctx,acqLauncher);

	TaskLauncher loadLauncher(CREATE_INTERFACE_TASK_ID, TaskArgument(&co,sizeof(co)));	// Spawn the renderer task
	loadLauncher.add_region_requirement(RegionRequirement(regions[0].get_logical_region(),READ_ONLY,EXCLUSIVE,regions[0].get_logical_region()));
	loadLauncher.add_field(0,FID_META);		// Metadata as first region
	loadLauncher.add_region_requirement(RegionRequirement(regions[1].get_logical_region(),WRITE_DISCARD,EXCLUSIVE,regions[1].get_logical_region()));
	loadLauncher.add_field(1,FID_VAL);		// Output Image as second region
	loadLauncher.add_region_requirement(RegionRequirement(regions[2].get_logical_region(),READ_ONLY,EXCLUSIVE,regions[2].get_logical_region()));
	loadLauncher.add_field(2,FID_VAL);		// Input Data as third region
	runtime->execute_task(ctx,loadLauncher);


	ReleaseLauncher relLauncher(outputLogicalRegion,outputLogicalRegion,regions[1]);
	relLauncher.add_field(FID_VAL);
	relLauncher.add_arrival_barrier(co.barrier);
	runtime->issue_release(ctx,relLauncher);

	runtime->advance_phase_barrier(ctx,co.barrier);

}

void coherence_composite_task(const Task*task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime){
	compositeArguments co = *((compositeArguments*)task->args);
	int inputRegionCount = regions.size() - 2;

	runtime->advance_phase_barrier(ctx,co.barrier);

	TaskLauncher loadLauncher(COMPOSITE_TASK_ID, TaskArgument(&co,sizeof(co)));	// Spawn the renderer task
	loadLauncher.add_region_requirement(RegionRequirement(regions[0].get_logical_region(),READ_ONLY,EXCLUSIVE,regions[0].get_logical_region()));
	loadLauncher.add_field(0,FID_META);		// Metadata as first region
	loadLauncher.add_region_requirement(RegionRequirement(regions[1].get_logical_region(),WRITE_DISCARD,EXCLUSIVE,regions[1].get_logical_region()));
	loadLauncher.add_field(1,FID_VAL);		// Output Image as second region


	for(int i = 0; i < inputRegionCount; ++i){
		LogicalRegion outputLogicalRegion = task->regions[2+i].region;
		AcquireLauncher acqLauncher(outputLogicalRegion,outputLogicalRegion,regions[2+i]);
		acqLauncher.add_field(FID_VAL);
		acqLauncher.add_wait_barrier(co.barrier);
		runtime->issue_acquire(ctx,acqLauncher);

		loadLauncher.add_region_requirement(RegionRequirement(outputLogicalRegion,READ_ONLY,EXCLUSIVE,outputLogicalRegion));
		loadLauncher.add_field(2+i,FID_VAL);


		ReleaseLauncher relLauncher(outputLogicalRegion,outputLogicalRegion,regions[2+i]);
		relLauncher.add_field(FID_VAL);
		runtime->issue_release(ctx,relLauncher);
	}

	runtime->execute_task(ctx,loadLauncher);
}


void top_level_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		Context ctx, HighLevelRuntime *runtime){
	/**
	 * Main task and visualization loop. Remains active throughout the entire execution of the program and spawns all execution tasks.
	 */
#ifdef QT
	inArgs args = {0,NULL}; // Construct fake arguments for the QT Window
	Interact(args); 		// Begin the process of spawning Qt using the interface
	//	sleep(1);				// The Qt window needs time to start, remove if loading data
#endif
	unsigned int datx = 216;	// Manually set (for now) bounds of the volumetric data
	unsigned int daty = 320;
	unsigned int datz = 320;

	DataMgr* dataMgr = new DataMgr;					// Spawn Xin's data manager to load the volumetric data
	const char *volumeFilename = "vort_mag_2.dat"; 	// The current data file
	dataMgr->loadRawFile(volumeFilename, datx, daty, datz, sizeof(float)); // Manual parameters for the size and shape
	float *volume = (float*)dataMgr->GetData(); 	// Get a pointer to the loaded data in memory
	size_t dim[3];
	dataMgr->GetDataDim(dim);						// float check the dimensions
	assert(dim[0]==datx && dim[1]==daty && dim[2]==datz);
	Rect<1> dataBound = Rect<1>(0,datx*daty*datz-1);	// Indexing the region used to hold the data (linearized)
	IndexSpace dataIndexSpace = runtime->create_index_space(ctx, Domain::from_rect<1>(dataBound)); //Create the Index Space (1 index per voxel)
	FieldSpace dataFieldSpace = runtime->create_field_space(ctx);	// Simple field space
	{
		FieldAllocator allocator = runtime->create_field_allocator(ctx,dataFieldSpace);
		allocator.allocate_field(sizeof(float),FID_VAL);			// Only requires one field
	}
	LogicalRegion dataLogicalRegion = runtime->create_logical_region(ctx,dataIndexSpace,dataFieldSpace); // Create the Logical Region
	{																// Populate the region
		RegionRequirement req(dataLogicalRegion, WRITE_DISCARD, EXCLUSIVE, dataLogicalRegion); // Filling requirement
		req.add_field(FID_VAL);
		InlineLauncher dataInlineLauncher(req);						// Inline launchers are simple


		PhysicalRegion dataPhysicalRegion = runtime->map_region(ctx,dataInlineLauncher);	// Map to a physical region
		dataPhysicalRegion.wait_until_valid();						// Should be pretty fast

		RegionAccessor<AccessorType::Generic, float> dataAccessor = dataPhysicalRegion.get_field_accessor(FID_VAL).typeify<float>();
		// The GPU's tested with have much better single precision performance. If this is changed, the renderer needs to be modified, too
		int i = 0;
		for(GenericPointInRectIterator<1> pir(dataBound); pir; pir++){	// Step through the data and write to the physical region
			dataAccessor.write(DomainPoint::from_point<1>(pir.p),volume[i++]); // Same order as data: X->Y->Z
		}
		runtime->unmap_region(ctx,dataPhysicalRegion);					// Free up resources
	}

	// The data load by this point should have taken enough time for Qt to start as well

	int width = 500;	// Arbitrarily chosen to encourage X-Window performance
	int height = 500;
	Rect<1> imgBound(Point<1>(0),Point<1>(width*height*4-1));	// Inclusive range of pixels on the screen
#ifdef QT
	bool first = true;
	Movement lastmov = {0,0,0};	// Holds the previous state
	Movement blankmov;			// For comparison against the empty struct
	while(!getDone()){ 			// Main Loop
		Movement mov = getMovement();	// Call the interface to see what Qt is requesting
		if(!first && mov == lastmov){	// If there is no new data, don't run the compositor and keep checking
			continue;
		}
		else{							// Do render a new (or first) image
			first = false;
			if(mov==blankmov){			// In the case that Qt failed to start in time use an arbitrarily chosen default
				mov = {{-6.38241, 17.6822, -3048.59, 2946.61, -2.39135, 42.4653, 391.329, -347.545, -45.7328, -4.6882, -505.424, 517.369, 9.43732e-09, 5.39067e-09, -4.99999, 5},0.0};
			}
		}
		lastmov = mov; // Update the state
#else
	{ 			// Main Loop

		Movement mov = {{28.4255, -13.5181, -2575.58, 2489.31, 17.9698, 42.9971, -1046.02, 1035.71, -37.6435, 10.3176, -2703.06, 2633, -3.36747e-8, 4.78245e-9, -5, 5.00001},1.0};
		cout << "Creating Image" << endl;	// All print statements need to be removed eventually
#endif
		IndexSpace imgIndex = runtime->create_index_space(ctx, Domain::from_rect<1>(imgBound)); // Set up the final image region
		FieldSpace imgField = runtime->create_field_space(ctx);
		{
			FieldAllocator allocator = runtime->create_field_allocator(ctx,imgField);
			allocator.allocate_field(sizeof(float),FID_VAL);	// Use the VAL field value as well
		}
		LogicalRegion imgLogicalRegion = runtime->create_logical_region(ctx,imgIndex,imgField);



		int numFiles = 31;							// Choose to create two partitions of the data
		vector<Image> images;						// Array to hold the metadata values in
		int zindex = 0;								// Keep track of the partition number
		vector<LogicalRegion> imgLogicalRegions;
		for(int i = 0; i < numFiles; ++i){
			int zspan = (int)(datz/numFiles);		// Split the data long the X-dimension
			Image newimg;							// Create a metadata object to hold values
			newimg.width = width;					// This data gets sent to the renderer (necessary)
			newimg.height = height;					// 		This is total image Width and Height
			for(int j = 0; j < 16; ++j)
				newimg.invPVM[j] = mov.invPVM[j];	// Copy the transformation matrix over
			newimg.xmin = 0;						// Values for the extent of the render within the image
			newimg.xmax = width-1;					// 		Set to be the entire size for now
			newimg.ymin = 0;						//		Need to feed partition bounds into the modelview to get these
			newimg.ymax = height-1;
			newimg.partition = (DataPartition){0,(int)datx,0,(int)daty, zindex,i==numFiles-1 ? (int)datz : zindex+zspan+10}; // Define the data partitioning
			// newimg.partition = (DataPartition){0,(int)datx,0,(int)daty, (int)datz / 2, (int)datz};// zindex,i==numFiles-1 ? (int)datz : zindex+zspan+10}; 
//			newimg.partition = (DataPartition){0, (int)datx, 0,(int)daty,0,(int)datz}; //
			newimg.order = mov.xdat * (float)zindex;// Feed the partition value into the modelview to get the compositing order
			images.push_back(newimg);				// Add the metadata to the array
			zindex += zspan;						// Iterate index values
			imgLogicalRegions.push_back(runtime->create_logical_region(ctx,imgIndex,imgField));
		}
		sort(images.rbegin(),images.rend());		// Sort the metadata in reverse value of order


		Rect<1> metadataBound(Point<1>(0),Point<1>(numFiles-1));	// Set up an index space for the metadata
		IndexSpace taskIndex = runtime->create_index_space(ctx, Domain::from_rect<1>(metadataBound));
		FieldSpace metadataField = runtime->create_field_space(ctx);
		{
			FieldAllocator allocator = runtime->create_field_allocator(ctx,metadataField);
			allocator.allocate_field(sizeof(Image),FID_META);	// Classify it with the META field value
		}
		LogicalRegion metadataLogicalRegion = runtime->create_logical_region(ctx, taskIndex, metadataField);




		{	// Fill the metadata Logical Region with the previously generated metadata
			RegionRequirement req(metadataLogicalRegion,WRITE_DISCARD,EXCLUSIVE,metadataLogicalRegion);
			req.add_field(FID_META);
			InlineLauncher metadataLauncher(req);

			PhysicalRegion metadataPhysicalRegion = runtime->map_region(ctx,metadataLauncher);
			metadataPhysicalRegion.wait_until_valid();

			RegionAccessor<AccessorType::Generic, Image> accessFilename = metadataPhysicalRegion.get_field_accessor(FID_META).typeify<Image>();
			// Using the 'Image' struct type
			int i = 0;
			for(GenericPointInRectIterator<1> pir(metadataBound); pir; pir++)
				accessFilename.write(DomainPoint::from_point<1>(pir.p),images[i++]);	// Make sure to write in the sorted order

			runtime->unmap_region(ctx,metadataPhysicalRegion);		// Free up resources
		}


		compositeArguments co;		// Arguments for calling composite tasks
		co.width = width;			// Total image size
		co.height = height;
		co.mov = mov;				// Inverse PV Matrix for tagging image
		co.miny = 0;				// Image possible extent (in Y-Dimension)
		co.maxy = height-1;			// For first level, must be entire image

//		PhaseBarrier doneBarrier = runtime->create_phase_barrier(ctx,numFiles);
//		co.barrier = doneBarrier;
//
//		MustEpochLauncher epochLauncher;
		for(unsigned int i = 0; i < images.size(); ++i){
			Point<1> p(i);
			Rect<1> metadataSubBound(p,p);
			IndexSpace metadataSubIndex = runtime->create_index_space(ctx,Domain::from_rect<1>(metadataSubBound));
			LogicalRegion metadataSubLogicalRegion = runtime->create_logical_region(ctx,metadataSubIndex,metadataField);
			{
				RegionRequirement req(metadataSubLogicalRegion,WRITE_DISCARD,EXCLUSIVE,metadataSubLogicalRegion);
				req.add_field(FID_META);
				InlineLauncher metadataLauncher(req);
				PhysicalRegion metadataPhysicalRegion = runtime->map_region(ctx,metadataLauncher);
				metadataPhysicalRegion.wait_until_valid();
				RegionAccessor<AccessorType::Generic,Image> accessFilename = metadataPhysicalRegion.get_field_accessor(FID_META).typeify<Image>();
				accessFilename.write(DomainPoint::from_point<1>(p),images[i]);
				runtime->unmap_region(ctx,metadataPhysicalRegion);
			}

//			TaskLauncher loadLauncher(COHERENCE_SIMULATION_TASK_ID, TaskArgument(&co,sizeof(co)));	// Spawn the renderer task
			TaskLauncher loadLauncher(CREATE_INTERFACE_TASK_ID, TaskArgument(&co,sizeof(co)));
			loadLauncher.add_region_requirement(RegionRequirement(metadataSubLogicalRegion,READ_ONLY,EXCLUSIVE,metadataSubLogicalRegion));
			loadLauncher.add_field(0,FID_META);		// Metadata as first region
//			loadLauncher.add_region_requirement(RegionRequirement(imgLogicalRegions[i],WRITE_DISCARD,SIMULTANEOUS,imgLogicalRegions[i]));
			loadLauncher.add_region_requirement(RegionRequirement(imgLogicalRegions[i],WRITE_DISCARD,EXCLUSIVE,imgLogicalRegions[i]));
			loadLauncher.add_field(1,FID_VAL);		// Output Image as second region
			loadLauncher.add_region_requirement(RegionRequirement(dataLogicalRegion,READ_ONLY,EXCLUSIVE,dataLogicalRegion));
			loadLauncher.add_field(2,FID_VAL);		// Input Data as third region
			runtime->execute_task(ctx,loadLauncher);
//			epochLauncher.add_single_task(DomainPoint(i),loadLauncher);

		}


//		TaskLauncher compositeLauncher(COHERENCE_COMPOSITE_TASK_ID, TaskArgument(&co,sizeof(co))); // Only one task, so use TaskLauncher
		TaskLauncher compositeLauncher(COMPOSITE_TASK_ID, TaskArgument(&co,sizeof(co)));
		compositeLauncher.add_region_requirement(RegionRequirement(metadataLogicalRegion,READ_ONLY,EXCLUSIVE,metadataLogicalRegion));
		compositeLauncher.add_field(0,FID_META); 	// Metadata as first region
		compositeLauncher.add_region_requirement(RegionRequirement(imgLogicalRegion,WRITE_DISCARD,EXCLUSIVE,imgLogicalRegion));
		compositeLauncher.add_field(1,FID_VAL);		// Output Image as second region
		for(unsigned int i = 0; i < images.size(); ++i){
//			compositeLauncher.add_region_requirement(RegionRequirement(imgLogicalRegions[i],READ_ONLY,SIMULTANEOUS,imgLogicalRegions[i]));
			compositeLauncher.add_region_requirement(RegionRequirement(imgLogicalRegions[i],READ_ONLY,EXCLUSIVE,imgLogicalRegions[i]));
			compositeLauncher.add_field(2+i,FID_VAL);
		}
		runtime->execute_task(ctx,compositeLauncher);
//		epochLauncher.add_single_task(DomainPoint(1),compositeLauncher);
//		runtime->execute_must_epoch(ctx,epochLauncher);




		TaskLauncher displayLauncher(DISPLAY_TASK_ID, TaskArgument(&co,sizeof(co)));	// Spawn a task for sending to Qt
		displayLauncher.add_region_requirement(RegionRequirement(imgLogicalRegion,READ_ONLY,EXCLUSIVE,imgLogicalRegion));
		displayLauncher.add_field(0,FID_VAL);	// Only needs the image (will map once compositor is done)

		runtime->execute_task(ctx,displayLauncher); // Run the display Task

	}
}








CompositeMapper::CompositeMapper(Machine m, HighLevelRuntime *rt, Processor p) : DefaultMapper(m, rt, p){
	/**
	 * Mapper for the compositor and renderer (will need to be modified for in-situ)
	 */
	set<Processor> all_procs;					// Prepare for the set of all processors available
	machine.get_all_processors(all_procs);		// Populate set
	top_proc = p;								// Current processor as top processor

	set<Processor>::iterator iter = all_procs.begin();	// Step through all processors
	iter++;												// Skip the first one (used for main loop)
	for(iter++; iter != all_procs.end();iter++){		// Add rest to a list of available processors for mapping
		task_procs.insert(*iter);
	}


	for (std::set<Processor>::const_iterator it = all_procs.begin(); it != all_procs.end(); it++){
		Processor::Kind k = it->kind();	// Differentiate CPU and GPU processors
		switch (k){
		case Processor::LOC_PROC:		// If CPU (Latency Optimized Core)
			all_cpus.push_back(*it);	// Add to CPU List
			break;
		case Processor::TOC_PROC:		// If GPU (Throughput Optimized Core)
			all_gpus.push_back(*it);	// Add to GPU List
			break;
		default:						// Something else...?
			break;
		}
	}
	{
		for (std::vector<Processor>::iterator itr = all_cpus.begin(); itr != all_cpus.end(); ++itr){
			Memory sysmem = machine_interface.find_memory_kind(*itr, Memory::SYSTEM_MEM);	// Find the relevant memories
			all_sysmems[*itr] = sysmem;
		}
	}
}

void CompositeMapper::select_task_options(Task *task){
	/**
	 * Specify properties and location of tasks
	 */
	task->inline_task = false;	// All of these off
	task->spawn_task = false;
	task->map_locally = false;
	task->profile_task = false;
	task->task_priority = 0;	// Can be used to specify some execution order (TO DO)
	if(task->get_depth()==0){	// If we're on the top-level task
		task->target_proc = local_proc; // Define it to the local processor
	}
	else{ // Otherwise define tasks on all of the other processors to reduce blocking
		if(task->task_id == CREATE_TASK_ID || task->task_id == CREATE_ISOSURFACE_TASK_ID || task->task_id == RENDER_ISOSURFACE_TASK_ID){ // Map the GPU tasks onto the GPU, though
			task->target_proc = DefaultMapper::select_random_processor(task_procs, Processor::TOC_PROC, machine);
		}
		else{
			task->target_proc = DefaultMapper::select_random_processor(task_procs, Processor::LOC_PROC, machine);
		}
	}
}


void CompositeMapper::slice_domain(const Task *task, const Domain &domain, std::vector<DomainSplit> &slices){
	/**
	 * Define how to split up Index Launch tasks
	 */
	std::vector<Processor> split_set;	// Find all processors to split on
	for (unsigned idx = 0; idx < 2; idx++){ // Add the approriate number for a binary decomposition
		split_set.push_back(DefaultMapper::select_random_processor(task_procs, Processor::LOC_PROC, machine));
	}

	DefaultMapper::decompose_index_space(domain, split_set,1/*splitting factor*/, slices); // Split the index space on colors
	for (std::vector<DomainSplit>::iterator it = slices.begin(); it != slices.end(); it++){
		Rect<1> rect = it->domain.get_rect<1>(); // Step through colors and indicate recursion or not
		if (rect.volume() == 1) // Stop recursing when only one task remains
			it->recurse = false;
		else
			it->recurse = true;
	}
}

bool CompositeMapper::map_task(Task *task){
	/**
	 * Control memory mapping for each task
	 */
	if (task->task_id == CREATE_TASK_ID){ // If running on the GPU
		Memory fb_mem = machine_interface.find_memory_kind(task->target_proc,Memory::GPU_FB_MEM); // Get FrameBuffer Memories
		assert(fb_mem.exists()); // Make sure it is supported
		for (unsigned idx = 0; idx < task->regions.size(); idx++){ 	// Step through all regions
			task->regions[idx].target_ranking.push_back(fb_mem); 	//	and map them to the framebuffer memory
			task->regions[idx].virtual_map = false;
			task->regions[idx].enable_WAR_optimization = war_enabled;
			task->regions[idx].reduction_list = false;
			// Make everything SOA
			task->regions[idx].blocking_factor = task->regions[idx].max_blocking_factor;
		}
	}
	else if(task->task_id == CREATE_ISOSURFACE_TASK_ID || task->task_id == RENDER_ISOSURFACE_TASK_ID){
		Memory fb_mem = machine_interface.find_memory_kind(task->target_proc,Memory::GPU_FB_MEM); // Get FrameBuffer Memories
		Memory zc_mem = machine_interface.find_memory_kind(task->target_proc,Memory::Z_COPY_MEM);
		assert(fb_mem.exists()); // Make sure it is supported
		assert(zc_mem.exists());
		for (unsigned idx = 0; idx < task->regions.size()-1; idx++){ 	// Step through all regions
			task->regions[idx].target_ranking.push_back(fb_mem); 	//	and map them to the framebuffer memory
			task->regions[idx].virtual_map = false;
			task->regions[idx].enable_WAR_optimization = war_enabled;
			task->regions[idx].reduction_list = false;
			// Make everything SOA
			task->regions[idx].blocking_factor = task->regions[idx].max_blocking_factor;
		}
		task->regions[3].target_ranking.push_back(zc_mem); 	//	and map them to the framebuffer memory
		task->regions[3].virtual_map = false;
		task->regions[3].enable_WAR_optimization = war_enabled;
		task->regions[3].reduction_list = false;
		// Make everything SOA
		task->regions[3].blocking_factor = task->regions[3].max_blocking_factor;

	}	
	else{
		// Put everything else in the system memory
		Memory sys_mem = all_sysmems[task->target_proc];
		assert(sys_mem.exists());
		for (unsigned idx = 0; idx < task->regions.size(); idx++)
		{
			task->regions[idx].target_ranking.push_back(sys_mem);
			task->regions[idx].virtual_map = false;
			task->regions[idx].enable_WAR_optimization = war_enabled;
			task->regions[idx].reduction_list = false;
			// Make everything SOA
			task->regions[idx].blocking_factor = task->regions[idx].max_blocking_factor;
		}
	}
	return false;
}

bool CompositeMapper::map_inline(Inline *inline_operation){
	bool ret = DefaultMapper::map_inline(inline_operation); // Call the default mapper version of this function
	RegionRequirement& req = inline_operation->requirement;
	req.blocking_factor = req.max_blocking_factor;	// But overwrite the blocking factor to force SOA
	return ret;
}


void mapper_registration(Machine machine, HighLevelRuntime *rt, const std::set<Processor> &local_procs){
	/**
	 * Register this mapper for each processor
	 */
	for (std::set<Processor>::const_iterator it = local_procs.begin(); it != local_procs.end(); it++){
		rt->replace_default_mapper(new CompositeMapper(machine, rt, *it), *it); // Step through all processors and create a mapper instance
	}
}


int main(int argc, char **argv){
	HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID); // Declare the main top-level task ID
	HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID, 	// Register the top-level task with legion
			Processor::LOC_PROC, true/*single*/, false/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(), "top_level");
	HighLevelRuntime::register_legion_task<coherence_simulation_task>(COHERENCE_SIMULATION_TASK_ID, 	// Register the simulation task
			Processor::LOC_PROC, true/*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(), "coherence_simulation_task");
	HighLevelRuntime::register_legion_task<coherence_composite_task>(COHERENCE_COMPOSITE_TASK_ID, 	// Register the simulation task
				Processor::LOC_PROC, true/*single*/, true/*index*/,
				AUTO_GENERATE_ID, TaskConfigOptions(), "coherence_composite_task");
	HighLevelRuntime::register_legion_task<composite_task>(COMPOSITE_TASK_ID, 	// Register the composite task (Inner Task)
			Processor::LOC_PROC, true/*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(false, false), "composite_task");
	HighLevelRuntime::register_legion_task<combine_task>(COMBINE_TASK_ID,		// Register combination task (Leaf Task)
			Processor::LOC_PROC, true/*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(true, false), "combine_task");
	HighLevelRuntime::register_legion_task<display_task>(DISPLAY_TASK_ID,		// Register Qt Display connection task (Leaf Task)
			Processor::LOC_PROC, true/*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(true, false), "display_task");
	HighLevelRuntime::register_legion_task<create_interface_task>(CREATE_INTERFACE_TASK_ID,		// Register Qt Display connection task (Leaf Task)
			Processor::LOC_PROC, true/*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(false, false), "create_interface_task");
	HighLevelRuntime::register_legion_task<create_isosurface_task>(CREATE_ISOSURFACE_TASK_ID,// Register the GPU render task (Leaf Task, TOC processor)
			Processor::TOC_PROC, true/*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(false, true), "create_isosurface_task");
	HighLevelRuntime::register_legion_task<create_task>(CREATE_TASK_ID,			// Register the GPU render task (Leaf Task, TOC processor)
			Processor::TOC_PROC, true/*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(false, true), "create_task");
	HighLevelRuntime::register_legion_task<render_isosurface_task>(RENDER_ISOSURFACE_TASK_ID,// Register the GPU render task (Leaf Task, TOC processor)
			Processor::TOC_PROC, true/*single*/, true/*index*/,
			AUTO_GENERATE_ID, TaskConfigOptions(false, true), "render_isosurface_task");
	HighLevelRuntime::set_registration_callback(mapper_registration);			// Register the custom mapper
	return HighLevelRuntime::start(argc, argv);									// Start the Legion runtime
}
