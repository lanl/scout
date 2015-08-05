/**
 * Ian Sohl - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Main Header
 */

#ifndef COMPOSITE_H
#define COMPOSITE_H


//#define ISOSURFACE
#define QT

#ifdef QT
#include "interact.h"
#endif
#include "legion.h"
#include "default_mapper.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

enum TaskIDs {
	TOP_LEVEL_TASK_ID,
	COHERENCE_SIMULATION_TASK_ID,
	COHERENCE_COMPOSITE_TASK_ID,
	COMPOSITE_TASK_ID,
	COMBINE_TASK_ID,
	LOAD_TASK_ID,
	DISPLAY_TASK_ID,
	CREATE_TASK_ID,
	CREATE_ISOSURFACE_TASK_ID,
	RENDER_ISOSURFACE_TASK_ID,
	CREATE_INTERFACE_TASK_ID,
}; /**<  List of task identifiers for compositing  */

enum FieldIDs {
	FID_META,
	FID_VAL,
	FID_VERTEX,
	FID_NORMAL,
	FID_NTRI,	//number of triangles
}; /**< List of field identifiers */

#ifndef QT
struct Movement{
	float invPVM[16]; /**< Inverse PV Matrix for rendering */
	float xdat;		  /**< X[3] value for composition ordering */

	bool operator==( const Movement& rhs ) const {
		/**
		 * Manually check for invPVM equivalence
		 */
		for(int i = 0; i < 16; ++i){
			if(abs(invPVM[i]-rhs.invPVM[i])>0.000001){ // Floating point problems
				return false;
			}
		}
		return true;
	}

	Movement& operator =(const Movement& a){
		/**
		 * Manual assignment
		 */
		for(int i = 0; i < 16; ++i){
			invPVM[i] = a.invPVM[i];
		}
		xdat = a.xdat;
	    return *this;
	}
}; /**< Current data state */

#endif

struct compositeArguments{
	int width;
	int height;
	Movement mov;
	int miny;
	int maxy;
	PhaseBarrier barrier;
}; /**< Arguments needed by compositing control tasks */

struct tripleArgument{
	compositeArguments co;
	compositeArguments co1;
	compositeArguments co2;
}; /**< Arguments needed by combination tasks */

struct DataPartition{
	int xmin;
	int xmax;
	int ymin;
	int ymax;
	int zmin;
	int zmax;
}; /**< Volumetric data partition bounding */

struct Image{
	int width;
	int height;
	float invPVM[16];
	int xmin;
	int xmax;
	int ymin;
	int ymax;
	DataPartition partition;
	float order;

	bool operator<( const Image& rhs ) const
	{ return order < rhs.order; }
}; /**< Individual image metadata structure (One per leaf render) */


class CompositeMapper : public DefaultMapper {
public:
	CompositeMapper(Machine machine, HighLevelRuntime *rt, Processor local);
public:
	virtual void select_task_options(Task *task);
	virtual void slice_domain(const Task *task, const Domain &domain, std::vector<DomainSplit> &slices);
	virtual bool map_task(Task *task);
	virtual bool map_inline(Inline *inline_operation);
protected:
	Processor top_proc;
	std::set<Processor> task_procs;
	std::map<Processor, Memory> all_sysmems;
	std::vector<Processor> all_cpus;
	std::vector<Processor> all_gpus;
};




#endif
