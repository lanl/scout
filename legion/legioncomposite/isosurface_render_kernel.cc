/**
 * Ian Sohl & Xin Tong - 2015
 * Copyright (c) 2015      Los Alamos National Security, LLC
 *                         All rights reserved.
 * Legion Image Composition - Isosurface Rendering Code
 */

#ifndef ISOSURFACE_RENDER_CC
#define ISOSURFACE_RENDER_CC

#include "composite.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_helper.h"
#include <optixu/optixpp_namespace.h>

#include <optix.h>
#include <optixu.h>
#include <optixu/optixu_aabb_namespace.h>
#include "optixu_vector_types.h"
#include "optixu_math_namespace.h"

#include "AccelDescriptor.h"
#include "commonStructs.h"
#include "radical_inverse_jitter.h"

#include <iostream>
using namespace optix;
using namespace std;

#define USE_PBO 0

optix::Context m_context;


//remember to set the running environment the build/ directory
const char* const ptxpath( const std::string& target, const std::string& base )
{
  static std::string path;
  path = "./" + target + "" + base + ".ptx";
  return path.c_str();
}



Buffer createOutputBuffer( RTformat format,
                                        unsigned int width,
                                        unsigned int height, float4* d_output)
{
  // Set number of devices to be used
  // Default, 0, means not to specify them here, but let OptiX use its default behavior.
//  if(m_num_devices)
//  {
//    int max_num_devices    = Context::getDeviceCount();
//    int actual_num_devices = std::min( max_num_devices, std::max( 1, m_num_devices ) );
//    std::vector<int> devs(actual_num_devices);
//    for( int i = 0; i < actual_num_devices; ++i ) devs[i] = i;
//    m_context->setDevices( devs.begin(), devs.end() );
//  }

#if USE_PBO
    // First allocate the memory for the GL buffer, then attach it to OptiX.
    //  initPBO();

    Buffer buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, pbo);
    buffer->setFormat(format);
    buffer->setSize( width, height );
#else
    // if(d_output != NULL)
    //     cudaFree(d_output);
    // if(cudaMalloc((void**) &d_output, sizeof(float4) * width * height) != cudaSuccess) {
    //     std::cout<<"memory allocation failed..."<<std::endl;
    //     exit(2);
    // }
    int d;
    cudaGetDevice(&d);
    Buffer buffer = m_context->createBufferForCUDA(RT_BUFFER_OUTPUT, format, width, height);
    buffer->setDevicePointer(d, reinterpret_cast<CUdeviceptr>(d_output));
#endif

  return buffer;
}


void initOptix(int nTri, int winWidth, int winHeight, float3* d_vertex, float3* d_normal, float4* d_output) {
    m_context = optix::Context::create();

    // Setup context
    m_context->setRayTypeCount(3);
    m_context->setEntryPointCount(1);
    m_context->setStackSize(cbrt((float)nTri) * 3);

    m_context["radiance_ray_type"]->setUint(0u);
    m_context["shadow_ray_type"]->setUint(1u);
    m_context["ao_ray_type"]->setUint(2u);
    m_context["scene_epsilon"]->setFloat(1.e-2f);

    // Output buffer
    Buffer outputBuffer = createOutputBuffer(RT_FORMAT_FLOAT4, winWidth, winHeight, d_output);
    //Buffer outputBuffer = createOutputBuffer(RT_FORMAT_FLOAT4, winWidth, winHeight);
    m_context["output_buffer"]->set(outputBuffer);


    // Lights buffer
    BasicLight lights[] =
    { // Light at left top front
      { make_float3(0, 0, 2), make_float3(1.0f, 1.0f, 1.0f), 0 }
    };

    Buffer lightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
    lightBuffer->setFormat(RT_FORMAT_USER);
    lightBuffer->setElementSize(sizeof(BasicLight));
    lightBuffer->setSize(sizeof(lights) / sizeof(lights[0]));
    memcpy(lightBuffer->map(), lights, sizeof(lights));
    lightBuffer->unmap();
    m_context["lights"]->set(lightBuffer);

    // Ray generation program
    std::string ptx_path = ptxpath( "", "matrix_camera");
    Program ray_gen_program = m_context->createProgramFromPTXFile(ptx_path, "matrix_camera");
    m_context->setRayGenerationProgram(0, ray_gen_program);

    // Exception / miss programs
    m_context->setExceptionProgram(0, m_context->createProgramFromPTXFile(ptx_path, "exception"));
    m_context["bad_color"]->setFloat(1.0f, 1.0f, 0.0f);

    m_context->setMissProgram(0, m_context->createProgramFromPTXFile(ptxpath("", "constantbg"), "miss"));
    m_context["bg_color"]->setFloat(0.0f, 0.0f, 0.0f);

    // Procedural materials (marble, wood, onion, noise_cubed, voronoi, sphere, checker)
    // If a wrong --proc parameter is used this will fail.
    std::string closest_hit_func_name( std::string( "closest_hit_" ) + "surface" );
    Program closest_hit_program = m_context->createProgramFromPTXFile(ptxpath("", "mesh_hit"), closest_hit_func_name );
    Program any_hit_program     = m_context->createProgramFromPTXFile(ptxpath("", "mesh_hit"), "any_hit_shadow");
    Program any_hit_occlusion_program     = m_context->createProgramFromPTXFile(ptxpath("", "mesh_hit"), "any_hit_occlusion");

    Material material = m_context->createMaterial();
    material->setClosestHitProgram(0u, closest_hit_program);
    material->setAnyHitProgram(1u, any_hit_program);
    material->setAnyHitProgram(2u, any_hit_occlusion_program);
    // Load model
    GeometryGroup geometry_group = m_context->createGeometryGroup();

    std::vector<GeometryInstance> instances;

    int d;
    cudaGetDevice(&d);

    Buffer vbuffer = m_context->createBufferForCUDA( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, nTri * 3 );
    vbuffer->setDevicePointer(d, reinterpret_cast<CUdeviceptr>(d_vertex));

    Buffer nbuffer = m_context->createBufferForCUDA(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, nTri * 3 );
    nbuffer->setDevicePointer(d, reinterpret_cast<CUdeviceptr>(d_normal));

    Geometry mesh;
    // Create the mesh object
    mesh = m_context->createGeometry();
    const std::string ptx_path2 = ptxpath( "", "triangle_mesh" );
    mesh->setPrimitiveCount( nTri );
    mesh->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path2, "mesh_intersect" ) );
    mesh->setBoundingBoxProgram( m_context->createProgramFromPTXFile(  ptx_path2, "mesh_bounds" ) );

    mesh[ "vertex_buffer" ]->setBuffer( vbuffer );
    mesh[ "normal_buffer" ]->setBuffer( nbuffer );

      // Create the geom instance to hold mesh and material params
    GeometryInstance instance = m_context->createGeometryInstance( mesh, &material, &material+1 );
    instance->setGeometry( mesh );
    instances.push_back( instance );

    // Set up group
    geometry_group->setChildCount( static_cast<unsigned int>(instances.size()) );
    Acceleration accel;
    accel = m_context->createAcceleration("Bvh", "Bvh");

    geometry_group->setAcceleration( accel);//acceleration );
    accel->markDirty();

//    AccelDescriptor m_accel_desc;
//    Acceleration acceleration = m_context->createAcceleration(m_accel_desc.builder.c_str(), /*m_compact_mesh ? "BvhCompact" : */m_accel_desc.traverser.c_str() );
////    acceleration->setProperty("refit", m_accel_desc.refit.c_str());
////    acceleration->setProperty("refine", m_accel_desc.refine.c_str());
////    acceleration->setProperty("vertex_buffer_name", "vertex_buffer");
//    //acceleration->setProperty( "index_buffer_name", "vindex_buffer" );
//    geometry_group->setAcceleration( acceleration );
//    acceleration->markDirty();


    for ( unsigned int i = 0; i < instances.size(); ++i )
      geometry_group->setChild( i, instances[i] );

    m_context[ "top_object" ]->set( geometry_group );
    m_context[ "top_shadower" ]->set( geometry_group );

    float tmp[16];
    m_context["invPVM"]->setMatrix4x4fv(false, tmp);
    m_context["normalMatrix"]->setMatrix3x3fv(false, tmp);
    m_context["modelviewMatrix"]->setMatrix4x4fv(false, tmp);


    // Material
    m_context["specular_exp"]->setFloat(20.0f);

    m_context["frame_number"]->setUint(1);
    m_context["jitter"]->setFloat(0.0f, 0.0f, 0.0f, 0.0f);

//    RTsize org_stack_size = m_context->getStackSize();
//    std::cout<<"original stake size: " << org_stack_size <<std::endl;
//    m_context->setStackSize(org_stack_size * 2);

    /* Random seed buffer */
//    RTbuffer   seed_buffer;
//    RTvariable seed_buffer_var;
//    unsigned int* seeds;
//    void *temp = 0;
//    RT_CHECK_ERROR2( rtBufferCreate( m_context, RT_BUFFER_INPUT, &seed_buffer ) );
//    RT_CHECK_ERROR2( rtBufferSetFormat( seed_buffer, RT_FORMAT_UNSIGNED_INT ) );
//    RT_CHECK_ERROR2( rtBufferSetSize2D( seed_buffer, winWidth, winHeight ) );
//    RT_CHECK_ERROR2( rtBufferMap( seed_buffer, &temp ) );
//    seeds = (unsigned int*)temp;
//    for( i = 0; i < winWidth*winHeight; ++i )
//      seeds[i] = rand();
//    RT_CHECK_ERROR2( rtBufferUnmap( seed_buffer ) );
//    RT_CHECK_ERROR2( rtContextDeclareVariable( m_context, "rnd_seeds",
//                                               &seed_buffer_var) );
//    RT_CHECK_ERROR2( rtVariableSetObject( seed_buffer_var, seed_buffer ) );
    m_context["occlusion_distance"]->setFloat(100.0f);
    m_context["sqrt_occlusion_samples"]->setInt(10);

    // Prepare to run
    m_context->validate();
    m_context->compile();

    // initialized = true;
    std::cout<<"Initialization done..."<<std::endl;

}

// inline void GetInvPVM(float modelview[16], float projection[16], float invPVM[16])
// {
//     QMatrix4x4 q_modelview(modelview);
//     q_modelview = q_modelview.transposed();

//     QMatrix4x4 q_projection(projection);
//     q_projection = q_projection.transposed();

//     QMatrix4x4 q_invProjMulView = (q_projection * q_modelview).inverted();

//     q_invProjMulView.copyDataTo(invPVM);
// }

// inline void GetNormalMatrix(float modelview[16], float NormalMatrix[9])
// {
//     QMatrix4x4 q_modelview(modelview);
//     q_modelview = q_modelview.transposed();

//     q_modelview.normalMatrix().copyDataTo(NormalMatrix);
// }



void drawOptix(float invPVM[16], float invMV[16], float modelview[16], float normalMatrix[9], float4* d_output, bool cameraChanged) {
    static unsigned sequence = 1;
    static unsigned frame = 0;

    if( cameraChanged ) {
      frame = 0;
      sequence = 1;
      cameraChanged = false;
    }

    // SaveMatrices(modelview, projection);
    // float invPVM[16];
    // float normalMatrix[9];
    //GetInvPVM(modelview, projection, invPVM);
    //GetNormalMatrix(modelview, normalMatrix);
    m_context["invPVM"]->setMatrix4x4fv(false, invPVM);
    m_context["normalMatrix"]->setMatrix3x3fv(false, normalMatrix);

    // QMatrix4x4 mv = QMatrix4x4(modelview);
    // mv = mv.transposed();
    // float invMV[16];
    // mv.inverted().copyDataTo(invMV);
    m_context["invModelviewMatrix"]->setMatrix4x4fv(false, invMV);
    m_context["modelviewMatrix"]->setMatrix4x4fv(false, modelview);

    //glUniformMatrix3fv(glProg->uniform("NormalMatrix"), 1, GL_FALSE, q_modelview.normalMatrix().data());



    RTsize buffer_width, buffer_height;
    m_context["output_buffer"]->getBuffer()->getSize(buffer_width, buffer_height);


    m_context["frame_number"]->setUint( ++frame );
//    std::cout<<"frame_number: "<<frame<<std::endl;
//    disp = (frame >= 33) || !((frame - 1) & 0x3); // Display frames 1, 5, 9, ... and 33+
    float4 jitter = radical_inverse_jitter( sequence );
//    std::cout<<"jitter: "<<jitter.x <<","<<jitter.y <<","<<jitter.z <<std::endl;
    m_context["jitter"]->setFloat( jitter );

    //the first time this function is called, it takes some time to build the acceleration structure
    m_context->launch(0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height) );

// #if !USE_PBO
//     cudaMemcpy(h_output, d_output, buffer_width * buffer_height * sizeof(float4), cudaMemcpyDeviceToHost);
// #endif
}

// void OptixPolyTracer::resize(int width, int height) {
//     Tracer::resize(width, height);
// #if USE_PBO
// #else

// //    //this following call slow down the frame rate
// //    Buffer outputBuffer = createOutputBuffer(RT_FORMAT_FLOAT4, winWidth, winHeight);
// //    m_context["output_buffer"]->set(outputBuffer);

// #endif
//     if(initialized) {
//         Buffer outputBuffer = createOutputBuffer(RT_FORMAT_FLOAT4, winWidth, winHeight);
//         //Buffer outputBuffer = createOutputBuffer(RT_FORMAT_FLOAT4, winWidth, winHeight);
//         m_context["output_buffer"]->set(outputBuffer);
//     }
// }

// void UpdateLightLoc(float x, float y)
// {
// //    std::cout<<"x: "<<x<<" y: "<<y<<std::endl;

//     // Lights buffer

//     float scale = 2;//dataMgr->GetMaxDim() ;

//     //the light position is set in object space, which is not entirely right.
//     BasicLight lights[] =
//     { // Light at left top front
//       { make_float3(x * scale, y * scale, scale), make_float3(1.0f, 1.0f, 1.0f), 0 }
//     };

//     Buffer lightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
//     lightBuffer->setFormat(RT_FORMAT_USER);
//     lightBuffer->setElementSize(sizeof(BasicLight));
//     lightBuffer->setSize(sizeof(lights) / sizeof(lights[0]));
//     memcpy(lightBuffer->map(), lights, sizeof(lights));
//     lightBuffer->unmap();
//     m_context["lights"]->set(lightBuffer);
// }



void render_isosurface_task(const Task *task,
		const std::vector<PhysicalRegion> &regions,
		LegionRuntime::HighLevel::Context ctx, HighLevelRuntime *runtime){

	compositeArguments co = *((compositeArguments*)task->args);
	int width = co.width;			// Get total image size
	int height = co.height;


	PhysicalRegion metadataPhysicalRegion = regions[0];
	LogicalRegion metadataLogicalRegion = metadataPhysicalRegion.get_logical_region();
	IndexSpace metadataIndexSpace = metadataLogicalRegion.get_index_space();
	Domain totalDomain = runtime->get_index_space_domain(ctx,metadataIndexSpace);
	Rect<1> totalRect = totalDomain.get_rect<1>();	// Get metadata value index


	RegionAccessor<AccessorType::Generic, Image> filenameAccessor = regions[0].get_field_accessor(FID_META).typeify<Image>();
	Image tmpimg = filenameAccessor.read(DomainPoint::from_point<1>(Point<1>(totalRect.lo.x[0])));	// Metadata for current render

	
	RegionAccessor<AccessorType::Generic, float> imgAccessor = regions[1].get_field_accessor(FID_VAL).typeify<float>(); // Accessor for data
	Domain imgDomain = runtime->get_index_space_domain(ctx,regions[1].get_logical_region().get_index_space());
	Rect<1> imgRect = imgDomain.get_rect<1>();	// Get data size domain
	Rect<1> imgSubRect;							// Empty filler rectangle
	ByteOffset imgOffsets[1];						// Byte Offset object
	float* imgPtr = imgAccessor.raw_rect_ptr<1>(imgRect,imgSubRect,imgOffsets); // Get raw framebuffer pointers
	
	
	
	//TODO: this boundary needs to be fixed in the real simulation
	// uint3 volstart = make_uint3(tmpimg.partition.xmin, tmpimg.partition.ymin, tmpimg.partition.zmin);
	// uint3 volend = make_uint3(tmpimg.partition.xmax,tmpimg.partition.ymax,tmpimg.partition.zmax);
	// uint3 slabsz = volend - volstart;
	// printf("volstart: %d, %d, %d  \n", volstart.x, volstart.y, volstart.z);
	// printf("volend: %d, %d, %d  \n", volend.x, volend.y, volend.z);

	// int chunkmaxverts = slabsz.x * slabsz.y * slabsz.z;
	// float3* v3f_chunk_d = GetDeviceArray<float3>(chunkmaxverts);
	// float3* n3f_chunk_d = GetDeviceArray<float3>(chunkmaxverts);
	
	
	
	Domain surfaceDomain = runtime->get_index_space_domain(ctx,regions[2].get_logical_region().get_index_space());
	Rect<1> surfaceRect = surfaceDomain.get_rect<1>();
	Rect<1> surfaceSubRect;
	ByteOffset surfaceOffsets[1];

	RegionAccessor<AccessorType::Generic, float3> VertexAccessor = regions[2].get_field_accessor(FID_VERTEX).typeify<float3>();
	float3* VertexPtr = VertexAccessor.raw_rect_ptr<1>(surfaceRect,surfaceSubRect,surfaceOffsets);
	RegionAccessor<AccessorType::Generic, float3> NormalAccessor = regions[2].get_field_accessor(FID_NORMAL).typeify<float3>();
	float3* NormalPtr = NormalAccessor.raw_rect_ptr<1>(surfaceRect,surfaceSubRect,surfaceOffsets);


	Domain ntriSurfaceDomain = runtime->get_index_space_domain(ctx,regions[3].get_logical_region().get_index_space());
	Rect<1> ntriSurfaceRect = ntriSurfaceDomain.get_rect<1>();
	Rect<1> ntriSurfaceSubRect;
	ByteOffset ntriSurfaceOffsets[1];
	RegionAccessor<AccessorType::Generic, int> ntriAccessor = regions[3].get_field_accessor(FID_NTRI).typeify<int>();
	//int* ntriPtr = ntriAccessor.raw_rect_ptr<1>(ntriSurfaceRect,ntriSurfaceSubRect,ntriSurfaceOffsets);
	int temp = ntriAccessor.read(DomainPoint::from_point<1>(Point<1>(0)));

	printf("number of triangles: %d \n", temp);
	initOptix(temp/*10083200*/, width, height, VertexPtr, NormalPtr, (float4*)imgPtr);

	float invPVM[] = {-6.38241, 17.6822, -3048.59, 2946.61, -2.39135, 42.4653, 391.329, -347.545, -45.7328, -4.6882, -505.424, 517.369, 9.43732e-09, 5.39067e-09, -4.99999, 5};
	float invMV[] = {-15.4085, 42.6886, 101.986, 609.719, -5.77322, 102.52, -43.7846, -78.266, -110.409, -11.3183, -11.9436, 101.085, -0, 0, -0, 1};
	float modelview[] = {-0.00123656, 0.00342582, 0.0081845, 0, -0.000463308, 0.0082274, -0.00351377, 0, -0.00886045, -0.000908309, -0.000958486, 0, 1.61335, -1.35304, -5.16837, 1};
	float normalMatrix[] = {48.8749, -3.35872, -100.304, 13.7393, 110.739, 2.98658, 99.4149, -13.6531, 48.8991};
	drawOptix(invPVM, invMV, modelview, normalMatrix, (float4*)imgPtr, true);

	//imgPtr[0] = 0 + tmpimg.partition.xmin;//NormalPtr[0].x + VertexPtr[0].x + slabsz.x;
	//CHECKME: I have no clue if this is the proper way to iterate
	// assert(chunkmaxverts==surfaceRect.volume())
	// for(int i = 0; i < chunkmaxverts; ++i){
	// 	v3f_chunk_d[i] = VertexPtr[i];
	// 	n3f_chunk_d[i] = NormalPtr[i];
	// }
	
	// for(int i = 0; i < width; i++) {
	// 	for(int j = 0; j < height; j++) {
	// 		imgPtr[i * height + j] = (i + j) * 0.0001;
	// 	}
	// }
	
	// Do isosurface render
	
}


#endif
