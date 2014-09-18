/**
 * Copyright (c) 2014      Los Alamos National Security, LLC
 *                         All rights reserved.
 */

#include "lsci.h"
#include "legion.h"

#include <iostream>
#include <functional>
#include <vector>
#include <iomanip>
#include <string>
#include <map>

#include <assert.h>
#include <stdio.h>

// convenience namespace aliases
namespace lrthl = LegionRuntime::HighLevel;

static size_t lsci_dt_size_tab[LSCI_TYPE_MAX + 1] = {
    sizeof(int32_t),
    sizeof(int64_t),
    sizeof(float),
    sizeof(double)
};

////////////////////////////////////////////////////////////////////////////////
// vector things
////////////////////////////////////////////////////////////////////////////////
int
lsci_vector_create(lsci_vector_t *vec,
                   size_t len,
                   lsci_dt_t type,
                   lsci_context_t context,
                   lsci_runtime_t runtime)
{
    using namespace LegionRuntime::HighLevel;
    using LegionRuntime::HighLevel::HighLevelRuntime;
    assert(vec && len > 0 && context && runtime && type < LSCI_TYPE_MAX);
    // first zero out everything in *vec
    (void)memset(vec, 0, sizeof(*vec));
    HighLevelRuntime *rtp_cxx = static_cast<HighLevelRuntime *>(runtime);
    Context *ctxp_cxx = static_cast<Context *>(context);
    vec->lr_len = len;
    vec->fid = 0;
    Rect<1> bounds = Rect<1>(Point<1>::ZEROES(), Point<1>(vec->lr_len - 1));
    // vector domain
    Domain dom(Domain::from_rect<1>(bounds));
    // vec index space
    IndexSpace *isp = new IndexSpace();
    *isp = rtp_cxx->create_index_space(*ctxp_cxx, dom);
    vec->index_space = static_cast<lsci_index_space_t>(isp);
    // vec field space
    FieldSpace fs = rtp_cxx->create_field_space(*ctxp_cxx);
    // vec field allocator
    FieldAllocator fa = rtp_cxx->create_field_allocator(*ctxp_cxx, fs);
    // all elements are going to be of size type
    fa.allocate_field(lsci_dt_size_tab[type], vec->fid);
    // now create the logical region
    LogicalRegion *lr = new LogicalRegion();
    *lr = rtp_cxx->create_logical_region(*ctxp_cxx, *isp, fs);
    vec->logical_region = static_cast<LogicalRegion *>(lr);
    // FIXME leak: add to C struct and free in lsci_vector_free
    //rtp_cxx->destroy_field_space(*ctxp_cxx, fs);
    return LSCI_SUCCESS;
}

int
lsci_vector_free(lsci_vector_t *vec,
                 lsci_context_t context,
                 lsci_runtime_t runtime)
{
    using namespace LegionRuntime::HighLevel;
    using LegionRuntime::HighLevel::HighLevelRuntime;
    assert(vec && context && runtime);

    HighLevelRuntime *rtp_cxx = static_cast<HighLevelRuntime *>(runtime);
    Context *ctxp_cxx = static_cast<Context *>(context);
    // if we had a logical partition, clean up resources
    if (vec->logical_partition) {
        LogicalPartition *lpp_cxx = static_cast<LogicalPartition *>(
                                        vec->logical_partition
                                    );
        rtp_cxx->destroy_logical_partition(*ctxp_cxx, *lpp_cxx);
        delete lpp_cxx;
    }
    IndexSpace *isp_cxx = static_cast<IndexSpace *>(vec->index_space);
    LogicalRegion *lrp_cxx = static_cast<LogicalRegion *>(
                                 vec->logical_region
                             );
    rtp_cxx->destroy_logical_region(*ctxp_cxx, *lrp_cxx);
    rtp_cxx->destroy_index_space(*ctxp_cxx, *isp_cxx);
    delete isp_cxx;
    delete lrp_cxx;
    return LSCI_SUCCESS;
}

int
lsci_vector_partition(lsci_vector_t *vec,
                      size_t n_parts,
                      lsci_context_t context,
                      lsci_runtime_t runtime)
{
    using namespace LegionRuntime::HighLevel;
    using LegionRuntime::HighLevel::HighLevelRuntime;
    using LegionRuntime::Arrays::Rect;
    assert(vec && n_parts > 0 && context && runtime);

    HighLevelRuntime *rtp_cxx = static_cast<HighLevelRuntime *>(runtime);
    Context *ctxp_cxx = static_cast<Context *>(context);
    IndexSpace *isp_cxx = static_cast<IndexSpace *>(vec->index_space);
    LogicalRegion *lrp_cxx = static_cast<LogicalRegion *>(vec->logical_region);

    const size_t gvlen = vec->lr_len;
    // FIXME for now, only allow even partitioning
    assert(0 == vec->lr_len % n_parts);
    size_t inc = gvlen / n_parts; // the increment
    Rect<1> colorBounds(Point<1>(0), Point<1>(n_parts - 1));
    Domain *colorDom = new Domain(Domain::from_rect<1>(colorBounds));
    //          +
    //          |
    //          |
    //     (x1)-+-+
    //          | |
    //          | m / nSubregions
    //     (x0) + |
    size_t x0 = 0, x1 = inc - 1;
    DomainColoring disjointColoring;
    // a list of sub-grid bounds.
    // provides a task ID to sub-grid bounds mapping.
    std::vector< Rect<1> > *subgrid_bnds_cxx = new std::vector< Rect<1> >;
    for (size_t color = 0; color < n_parts; ++color) {
        Rect<1> subRect((Point<1>(x0)), (Point<1>(x1)));
        // cache the subgrid bounds
        subgrid_bnds_cxx->push_back(subRect);
#if 0 // nice debug
        printf("vec disjoint partition: (%d) to (%d)\n",
                subgrid_bnds_cxx->at(color).lo.x[0],
                subgrid_bnds_cxx->at(color).hi.x[0]);
#endif
        disjointColoring[color] = Domain::from_rect<1>(subRect);
        x0 += inc;
        x1 += inc;
    }
    IndexPartition iPart = rtp_cxx->create_index_partition(
                               *ctxp_cxx, *isp_cxx,
                               *colorDom, disjointColoring,
                               true /* disjoint */
                           );
    // logical partitions
    using LegionRuntime::HighLevel::LogicalPartition;
    LogicalPartition *lpp_cxx = new LogicalPartition();
    *lpp_cxx = rtp_cxx->get_logical_partition(*ctxp_cxx, *lrp_cxx, iPart);
    vec->logical_partition = static_cast<lsci_logical_partition_t>(lpp_cxx);
    // launch domain -- one task per color
    vec->launch_domain.hndl = static_cast<lsci_domain_handle_t>(colorDom);
    vec->launch_domain.volume = colorDom->get_volume();
    vec->subgrid_bounds_len = vec->lr_len / n_parts;
    //printf("subgrid_bounds_len set: %lu\n", vec->subgrid_bounds_len);
    vec->subgrid_bounds = static_cast<lsci_rect_1d_t>(subgrid_bnds_cxx->data());
    return LSCI_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
size_t
lsci_sizeof_cxx_rect_1d(void)
{
    using namespace LegionRuntime::HighLevel;
    return sizeof(Rect<1>);
}

////////////////////////////////////////////////////////////////////////////////
lsci_rect_1d_t
lsci_subgrid_bounds_at(lsci_rect_1d_t rect_1d_array_basep,
                       size_t index)
{
    assert(rect_1d_array_basep && index >= 0);
    Rect<1> *pos = static_cast<Rect<1> *>(rect_1d_array_basep);
    pos += index;
    return pos;
}

////////////////////////////////////////////////////////////////////////////////
void
lsci_subgrid_bounds_at_set(lsci_rect_1d_t rect_1d_array_basep,
                       size_t index, lsci_rect_1d_storage_t* dest)
{
    assert(rect_1d_array_basep && index >= 0);
    Rect<1> *pos = static_cast<Rect<1> *>(rect_1d_array_basep);
    pos += index;
    *dest = *((lsci_rect_1d_storage_t*)pos);
}

////////////////////////////////////////////////////////////////////////////////
int
lsci_argument_map_create(lsci_argument_map_t *arg_map)
{
    using namespace LegionRuntime::HighLevel;
    assert(arg_map);
    ArgumentMap *arg_map_cxx = new ArgumentMap();
    assert(arg_map_cxx);
    arg_map->hndl = static_cast<lsci_argument_map_handle_t>(arg_map_cxx);
    return LSCI_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
int
lsci_argument_map_set_point(lsci_argument_map_t *arg_map,
                            size_t tid,
                            void *payload_base,
                            size_t payload_extent)
{
    using namespace LegionRuntime::HighLevel;
    assert(arg_map && payload_base);
    ArgumentMap *arg_map_cxx = static_cast<ArgumentMap *>(arg_map->hndl);
    assert(arg_map_cxx);
    arg_map_cxx->set_point(DomainPoint::from_point<1>(Point<1>(tid)),
                           TaskArgument(payload_base, payload_extent));
    return LSCI_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
int
lsci_index_launcher_create(lsci_index_launcher_t *il,
                           int task_id,
                           lsci_domain_t *ldom,
                           void *task_arg,
                           size_t task_arg_extent,
                           lsci_argument_map_t *arg_map)
{
    using namespace LegionRuntime::HighLevel;
    using LegionRuntime::HighLevel::HighLevelRuntime;
    assert(il && ldom); // task_arg and arg_map can be NULL
    assert(task_arg_extent >= 0);
    if (task_arg_extent > 0) assert(task_arg);

    Domain *ldom_cxx = static_cast<Domain *>(ldom->hndl);
    assert(ldom_cxx);
    ArgumentMap *arg_map_cxx = static_cast<ArgumentMap *>(arg_map->hndl);
    IndexLauncher *ilp_cxx = new IndexLauncher(
                                     task_id, *ldom_cxx,
                                     TaskArgument(task_arg, task_arg_extent),
                                     *arg_map_cxx
                                 );
    il->hndl = static_cast<lsci_index_launcher_handle_t>(ilp_cxx);
    return LSCI_SUCCESS;
}

int
lsci_add_region_requirement(lsci_index_launcher_t *il,
                            lsci_logical_region_t lr,
                            lsci_projection_id_t proj_id,
                            lsci_privilege_mode_t priv_mode,
                            lsci_coherence_property_t coherence_prop,
                            lsci_logical_partition_t parent)
{
    using namespace LegionRuntime::HighLevel;
    using LegionRuntime::HighLevel::HighLevelRuntime;
    assert(il);
    IndexLauncher *ilp_cxx = static_cast<IndexLauncher *>(il->hndl);
    LogicalPartition *lpp_cxx = static_cast<LogicalPartition *>(lr);
    LogicalRegion *parp_cxx = static_cast<LogicalRegion *>(parent);
    ilp_cxx->add_region_requirement(
        RegionRequirement(
            *lpp_cxx, proj_id,
            (PrivilegeMode)priv_mode,
            (CoherenceProperty)coherence_prop,
            *parp_cxx
        )
    );
    return LSCI_SUCCESS;
}

int
lsci_add_field(lsci_index_launcher_t *il,
               unsigned int idx,
               lsci_field_id_t field_id)
{
    using namespace LegionRuntime::HighLevel;
    using LegionRuntime::HighLevel::HighLevelRuntime;
    assert(il);

    IndexLauncher *ilp_cxx = static_cast<IndexLauncher *>(il->hndl);
    ilp_cxx->add_field(idx, field_id);
    return LSCI_SUCCESS;
}

int
lsci_execute_index_space(lsci_runtime_t runtime,
                         lsci_context_t context,
                         lsci_index_launcher_t *il)
{
    using namespace LegionRuntime::HighLevel;
    using LegionRuntime::HighLevel::HighLevelRuntime;
    assert(runtime && context && il);

    HighLevelRuntime *rtp_cxx = static_cast<HighLevelRuntime *>(runtime);
    Context *ctxp_cxx = static_cast<Context *>(context);
    IndexLauncher *ilp_cxx = static_cast<IndexLauncher *>(il->hndl);
    rtp_cxx->execute_index_space(*ctxp_cxx, *ilp_cxx);
    return LSCI_SUCCESS;
}

namespace {
int
reg_void_legion_task_cxx(
    const lrthl::Task *task,
    const std::vector<lrthl::PhysicalRegion> &rgns,
    lrthl::Context ctx,
    lrthl::HighLevelRuntime *lrt,
    const lsci_reg_task_data_t &cb_args
)
{
    using namespace LegionRuntime::HighLevel;

    lsci_task_args_t targs = {
        .context = static_cast<lsci_context_t>(&ctx),
        .runtime = static_cast<lsci_runtime_t>(lrt),
        .task = (lsci_task_t)task,
        .task_id = task->index_point.point_data[0],
        .n_regions = rgns.size(),
        .regions = (lsci_physical_regions_t)rgns.data(),
        .argsp = task->args,
        .local_argsp = task->local_args
    };
    /* now call the provided callback and pass it all the info it needs */
    cb_args.cbf(&targs);
    return LSCI_SUCCESS;
}

template <unsigned DIM, typename T>
bool
offsetsAreDense(const Rect<DIM> &bounds,
                const LegionRuntime::Accessor::ByteOffset *offset)
{
    off_t exp_offset = sizeof(T);
    for (unsigned i = 0; i < DIM; i++) {
        bool found = false;
        for (unsigned j = 0; j < DIM; j++)
            if (offset[j].offset == exp_offset) {
                found = true;
                exp_offset *= (bounds.hi[j] - bounds.lo[j] + 1);
                break;
            }
        if (!found) return false;
    }
    return true;
}

} // end namespace

int
lsci_start(int argc,
           char **argv)
{
    using namespace LegionRuntime::HighLevel;
    // make sure that the compile-time constant is valid. if this triggers,
    // simply update LSCI_RECT_1D_CXX_SIZE's value. see comment in header.
    assert(LSCI_RECT_1D_CXX_SIZE == lsci_sizeof_cxx_rect_1d());
    // also make sure that the struct for storage is the same size
    assert(sizeof(lsci_rect_1d_storage_t) == lsci_sizeof_cxx_rect_1d());
    return HighLevelRuntime::start(argc, argv);
}

void
lsci_set_top_level_task_id(int task_id)
{
    using namespace LegionRuntime::HighLevel;

    HighLevelRuntime::set_top_level_task_id(task_id);
}

int
lsci_register_void_legion_task(
    int task_id,
    lsci_proc_kind_t p_kind,
    bool single,
    bool index,
    bool leaf,
    lsci_variant_id_t vid,
    char *name,
    lsci_reg_task_data_t reg_task_data
)
{
    using namespace LegionRuntime::HighLevel;

    HighLevelRuntime::register_legion_task
    <int, lsci_reg_task_data_t, reg_void_legion_task_cxx>(
        (TaskID)task_id,
        (Processor::Kind)p_kind,
        single,
        index,
        reg_task_data,
        vid,
        TaskConfigOptions(leaf /* leaf task */),
        name
    );
    return LSCI_SUCCESS;
}

int
lsci_register_void_legion_task_aux(
    int task_id,
    lsci_proc_kind_t p_kind,
    bool single,
    bool index,
    bool leaf,
    lsci_variant_id_t vid,
    char *name,
    void (*atask)(struct lsci_task_args_t *task_args))
{

  lsci_reg_task_data_t* reg_task_data_ptr = new lsci_reg_task_data_t;
  reg_task_data_ptr->cbf = atask;

  return lsci_register_void_legion_task(
      task_id,
      p_kind,
      single,
      index,
      leaf,
      vid,
      name,
      *reg_task_data_ptr);
}


namespace {
template<typename T>
void
dump(const LegionRuntime::HighLevel::LogicalRegion &lr,
     LegionRuntime::HighLevel::FieldID fid,
     const LegionRuntime::Arrays::Rect<1> &bounds,
     std::string prefix,
     size_t nle,
     LegionRuntime::HighLevel::Context &ctx,
     LegionRuntime::HighLevel::HighLevelRuntime *lrt)
{
    using namespace LegionRuntime::HighLevel;
    using namespace LegionRuntime::Accessor;
    using LegionRuntime::Arrays::Rect;

    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(fid);
    InlineLauncher dumpl(req);
    PhysicalRegion reg = lrt->map_region(ctx, dumpl);
    reg.wait_until_valid();
    typedef RegionAccessor<AccessorType::Generic, T> GTRA;
    GTRA acc = reg.get_field_accessor(fid).typeify<T>();
    typedef GenericPointInRectIterator<1> GPRI1D;
    typedef DomainPoint DomPt;
    std:: cout << "*** " << prefix << " ***" << std::endl;
    int i = 0;
    for (GPRI1D pi(bounds); pi; pi++, ++i) {
        T val = acc.read(DomPt::from_point<1>(pi.p));
        if (i % nle == 0) {
            std::cout << std::endl;
        }
        std::cout << std::setfill(' ') << std::setw(3) << val << " ";
    }
}
}

int
lsci_vector_dump(lsci_vector_t *vec,
                 lsci_dt_t type,
                 lsci_context_t context,
                 lsci_runtime_t runtime)
{
    using namespace LegionRuntime::HighLevel;
    using namespace LegionRuntime::Accessor;
    using LegionRuntime::Arrays::Rect;
    assert(vec);

    HighLevelRuntime *rtp_cxx = static_cast<HighLevelRuntime *>(runtime);
    Context *ctxp_cxx = static_cast<Context *>(context);
    LogicalRegion *lrp_cxx = static_cast<LogicalRegion *>(vec->logical_region);
    FieldID fid_cxx = static_cast<FieldID>(vec->fid);

    switch (type) {
        case LSCI_TYPE_INT32: {
            dump<int32_t>(*lrp_cxx, fid_cxx,
                          Rect<1>(Point<1>::ZEROES(), Point<1>(vec->lr_len - 1)),
                          "int32 dump", 32, *ctxp_cxx, rtp_cxx);
            break;
        }
        case LSCI_TYPE_INT64: {
            dump<int64_t>(*lrp_cxx, fid_cxx,
                          Rect<1>(Point<1>::ZEROES(), Point<1>(vec->lr_len - 1)),
                          "int64 dump", 32, *ctxp_cxx, rtp_cxx);
            break;
        }
        case LSCI_TYPE_FLOAT: {
            dump<float>(*lrp_cxx, fid_cxx,
                         Rect<1>(Point<1>::ZEROES(), Point<1>(vec->lr_len - 1)),
                         "float dump", 32, *ctxp_cxx, rtp_cxx);
            break;
        }
        case LSCI_TYPE_DOUBLE: {
            dump<double>(*lrp_cxx, fid_cxx,
                         Rect<1>(Point<1>::ZEROES(), Point<1>(vec->lr_len - 1)),
                         "double dump", 32, *ctxp_cxx, rtp_cxx);
            break;
        }
        default:
            assert(false && "invalid lsci_dt_t");
    }

    return LSCI_SUCCESS;
}

void *
lsci_raw_rect_ptr_1d(lsci_physical_regions_t rgnp,
                     lsci_dt_t type,
                     size_t rid,
                     lsci_field_id_t fid,
                     lsci_task_t task,
                     lsci_context_t context,
                     lsci_runtime_t runtime)
{
    using namespace LegionRuntime::HighLevel;
    using namespace LegionRuntime::Accessor;

    assert(context && runtime && rgnp && type < LSCI_TYPE_MAX);
    assert(rgnp && rid >= 0);
    HighLevelRuntime *rtp_cxx = static_cast<HighLevelRuntime *>(runtime);
    Task *taskp_cxx = static_cast<Task *>(task);
    Context *ctxp_cxx = static_cast<Context *>(context);
    // get the base of the PhysicalRegion array
    PhysicalRegion *prgnp_cxx = static_cast<PhysicalRegion *>(rgnp);
    // index into the array given the region id
    prgnp_cxx += rid;
    // get the domain so we can get the sub-grid bounds later
    Domain dom_cxx = rtp_cxx->get_index_space_domain(
                         *ctxp_cxx,
                         taskp_cxx->regions[rid].region.get_index_space()
                     );
    Rect<1> sgb_cxx = dom_cxx.get_rect<1>();

    void *resultp = NULL;
    Rect<1> subRect;
    ByteOffset bOff[1];

    switch (type) {
        case LSCI_TYPE_INT32: {
            typedef RegionAccessor<AccessorType::Generic, int32_t> RA;
            RA fm = prgnp_cxx->get_field_accessor(fid).typeify<int32_t>();
            resultp = fm.raw_rect_ptr<1>(sgb_cxx, subRect, bOff);
            if (!resultp || (subRect != sgb_cxx) ||
                !offsetsAreDense<1, int32_t>(sgb_cxx, bOff)) {
                assert(false && "Cannot continue >:-|");
            }
            return resultp;
        }
        case LSCI_TYPE_INT64: {
            typedef RegionAccessor<AccessorType::Generic, int64_t> RA;
            RA fm = prgnp_cxx->get_field_accessor(fid).typeify<int64_t>();
            resultp = fm.raw_rect_ptr<1>(sgb_cxx, subRect, bOff);
            if (!resultp || (subRect != sgb_cxx) ||
                !offsetsAreDense<1, int64_t>(sgb_cxx, bOff)) {
                assert(false && "Cannot continue >:-|");
            }
            return resultp;
        }
        case LSCI_TYPE_FLOAT: {
            typedef RegionAccessor<AccessorType::Generic, float> RA;
            RA fm = prgnp_cxx->get_field_accessor(fid).typeify<float>();
            resultp = fm.raw_rect_ptr<1>(sgb_cxx, subRect, bOff);
            if (!resultp || (subRect != sgb_cxx) ||
                !offsetsAreDense<1, float>(sgb_cxx, bOff)) {
                assert(false && "Cannot continue >:-|");
            }
            return resultp;
        }
        case LSCI_TYPE_DOUBLE: {
            typedef RegionAccessor<AccessorType::Generic, double> RA;
            RA fm = prgnp_cxx->get_field_accessor(fid).typeify<double>();
            resultp = fm.raw_rect_ptr<1>(sgb_cxx, subRect, bOff);
            if (!resultp || (subRect != sgb_cxx) ||
                !offsetsAreDense<1, double>(sgb_cxx, bOff)) {
                assert(false && "Cannot continue >:-|");
            }
            return resultp;
        }
        default:
            assert(false && "invalid lsci_dt_t");
    }
    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// mesh things
////////////////////////////////////////////////////////////////////////////////
namespace {
struct mesh_cxx {
    // set based on w, h, and d
    size_t dims;
    size_t width;
    size_t height;
    size_t depth;
    // total number of elements stored in mesh
    size_t nelems;
    // FIXME map leak. Plug in free
    std::map<std::string, lsci_vector_t> vectab;

    mesh_cxx(void) {
        dims   = 0;
        width  = 0;
        height = 0;
        depth  = 0;
        nelems = 0;
    }

    mesh_cxx(size_t width,
             size_t height,
             size_t depth) :
        width(width), height(height), depth(depth) {
        assert(width > 0);
        set_dims();
        set_nelemes();
    }

    void
    free(lsci_context_t context,
         lsci_runtime_t runtime) {
        assert(context && runtime);
        typedef std::map<std::string, lsci_vector_t>::iterator MapI;
        for (MapI i = vectab.begin(); i != vectab.end(); i++) {
            lsci_vector_free(&i->second, context, runtime);
        }
    }

    void
    add_field(std::string name, lsci_vector_t vec) {
        assert(vectab.find(name) == vectab.end() && "duplicate field");
        vectab[name] = vec;
    }

    lsci_vector_t
    get_field(std::string name) {
        typedef std::map<std::string, lsci_vector_t>::iterator MapI;
        MapI fi = vectab.find(name);
        assert(fi != vectab.end() && "field not found");
        return fi->second;
    }

    void
    partition(size_t n_parts,
              lsci_context_t context,
              lsci_runtime_t runtime) {
        assert(n_parts && context && runtime);
        typedef std::map<std::string, lsci_vector_t>::iterator MapI;
        for (MapI i = vectab.begin(); i != vectab.end(); i++) {
            std::cout << "lsci: partitioning: " << i->first <<
                         " into " << n_parts << " piece(s)." << std::endl;
            lsci_vector_partition(&i->second, n_parts, context, runtime);
        }
    }

private:
    void
    set_dims(void) {
        dims = 1;
        if (height > 1) dims++;
        if (depth > 1) dims++;
    }

    void
    set_nelemes(void) {
        nelems = width;
        if (dims >= 2) nelems *= height;
        if (dims >= 3) nelems *= depth;
        if (dims > 3) assert(false && "not supported");
    }
};

} // end unnamed namespace for internal mesh things

int
lsci_unimesh_create(lsci_unimesh_t *mesh,
                    size_t w,
                    size_t h,
                    size_t d,
                    lsci_context_t context,
                    lsci_runtime_t runtime)
{
    assert(mesh && context && runtime && w);
    mesh_cxx *mcxx = new mesh_cxx(w, h, d);
    mesh->hndl = static_cast<lsci_unimesh_handle_t>(mcxx);
    assert(mesh->hndl);
    mesh->dims = mcxx->dims;
    mesh->width = w;
    mesh->height = h;
    mesh->depth = d;
    return LSCI_SUCCESS;
}

int
lsci_unimesh_free(lsci_unimesh_t *mesh,
                  lsci_context_t context,
                  lsci_runtime_t runtime)
{
    assert(mesh && context && runtime);
    mesh_cxx *mp_cxx = static_cast<mesh_cxx *>(mesh->hndl);
    mp_cxx->free(context, runtime);
    return LSCI_SUCCESS;
}

int
lsci_unimesh_add_field(lsci_unimesh_t *mesh,
                       lsci_dt_t type,
                       char *field_name,
                       lsci_context_t context,
                       lsci_runtime_t runtime)
{
    assert(mesh && field_name && context && runtime && type < LSCI_TYPE_MAX);
    mesh_cxx *mcxx = static_cast<mesh_cxx *>(mesh->hndl);
    assert(mcxx);
    lsci_vector_t field;
    lsci_vector_create(&field, mcxx->nelems, type, context, runtime);
    // now add the thing to the map
    mcxx->add_field(std::string(field_name), field);
    return LSCI_SUCCESS;
}

int
lsci_unimesh_partition(lsci_unimesh_t *mesh,
                       size_t n_parts,
                       lsci_context_t context,
                       lsci_runtime_t runtime)
{
    assert(mesh && n_parts && context && runtime);
    mesh_cxx *mcxx = static_cast<mesh_cxx *>(mesh->hndl);
    assert(mcxx);
    mcxx->partition(n_parts, context, runtime);
    return LSCI_SUCCESS;
}

int
lsci_unimesh_get_vec_by_name(lsci_unimesh_t *mesh,
                             char *name,
                             lsci_vector_t *vec,
                             lsci_context_t context,
                             lsci_runtime_t runtime)
{
    assert(mesh && name && vec && context && runtime);
    mesh_cxx *mcxx = static_cast<mesh_cxx *>(mesh->hndl);
    assert(mcxx);
    *vec = mcxx->get_field(std::string(name));
    return LSCI_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
// struct things
////////////////////////////////////////////////////////////////////////////////
namespace {
struct struct_cxx {
    std::map<std::string, lsci_vector_t> vectab;

    struct_cxx(void) {

    }

    void
    add_field(std::string name, lsci_vector_t vec) {
        assert(vectab.find(name) == vectab.end() && "duplicate field");
        vectab[name] = vec;
    }

    lsci_vector_t
    get_field(std::string name) {
        typedef std::map<std::string, lsci_vector_t>::iterator MapI;
        MapI fi = vectab.find(name);
        assert(fi != vectab.end() && "field not found");
        return fi->second;
    }

    void
    partition(size_t n_parts,
              lsci_context_t context,
              lsci_runtime_t runtime) {
        assert(n_parts && context && runtime);
        typedef std::map<std::string, lsci_vector_t>::iterator MapI;
        for (MapI i = vectab.begin(); i != vectab.end(); i++) {
            std::cout << "lsci: partitioning: " << i->first <<
                         " into " << n_parts << " piece(s)." << std::endl;
            lsci_vector_partition(&i->second, n_parts, context, runtime);
        }
    }
};

} // end unnamed namespace for internal things

int
lsci_struct_create(lsci_struct_t *theStruct,
                   lsci_context_t context,
                   lsci_runtime_t runtime)
{
    assert(theStruct && context && runtime);
    struct_cxx *scxx = new struct_cxx;
    theStruct->hndl = static_cast<lsci_struct_handle_t>(scxx);
    assert(theStruct->hndl);
    return LSCI_SUCCESS;
}

int
lsci_struct_add_field(lsci_struct_t *theStruct,
                      lsci_dt_t type,
                      size_t length,
                      char *field_name,
                      lsci_context_t context,
                      lsci_runtime_t runtime)
{
    assert(theStruct && field_name &&
           context && runtime &&
           type < LSCI_TYPE_MAX);
    struct_cxx *scxx = static_cast<struct_cxx *>(theStruct->hndl);
    assert(scxx);
    lsci_vector_t field;
    lsci_vector_create(&field, length, type, context, runtime);
    // now add the thing to the map
    scxx->add_field(std::string(field_name), field);
    return LSCI_SUCCESS;
}

int
lsci_struct_partition(lsci_struct_t *theStruct,
                      size_t n_parts,
                      lsci_context_t context,
                      lsci_runtime_t runtime)
{
    assert(theStruct && n_parts && context && runtime);
    struct_cxx *scxx = static_cast<struct_cxx *>(theStruct->hndl);
    assert(scxx);
    scxx->partition(n_parts, context, runtime);
    return LSCI_SUCCESS;
}

int
lsci_struct_get_vec_by_name(lsci_struct_t *theStruct,
                            char *name,
                            lsci_vector_t *vec,
                            lsci_context_t context,
                            lsci_runtime_t runtime)
{
    assert(theStruct && name && vec && context && runtime);
    struct_cxx *scxx = static_cast<struct_cxx *>(theStruct->hndl);
    assert(scxx);
    *vec = scxx->get_field(std::string(name));
    return LSCI_SUCCESS;
}

int
lsci_get_index_space_domain(
    lsci_runtime_t runtime,
    lsci_context_t context,
    lsci_task_t task,
    size_t rid,
    lsci_domain_t *answer_bufp
)
{
    using namespace LegionRuntime::HighLevel;
    using LegionRuntime::HighLevel::HighLevelRuntime;

    assert(runtime && context && task && answer_bufp);

    HighLevelRuntime *rtp_cxx = static_cast<HighLevelRuntime *>(runtime);
    Context *ctxp_cxx = static_cast<Context *>(context);
    Task *taskp_cxx = static_cast<Task *>(task);
    assert(rtp_cxx && ctxp_cxx && taskp_cxx);

    Domain *domp_cxx = new Domain(rtp_cxx->get_index_space_domain(
                           *ctxp_cxx,
                           taskp_cxx->regions[rid].region.get_index_space()
                       ));
    answer_bufp->hndl = static_cast<lsci_domain_handle_t>(domp_cxx);
    answer_bufp->volume = domp_cxx->get_volume();
    return LSCI_SUCCESS;
}

void
lsci_print_mesh_task_args(lsci_mesh_task_args_t* mtargs)
{
    printf("lsci_mesh_task_args: \n");
    printf("\trank: %lu\n", mtargs->rank);
    printf("\twidth: %lu\n", mtargs->global_width);
    printf("\theight: %lu\n", mtargs->global_height);
    printf("\tdepth: %lu\n", mtargs->global_depth);
    printf("\tlen: %lu\n", mtargs->sgb_len);
}

void
lsci_print_task_args_local_argsp(lsci_task_args_t* targs)
{
    printf("lsci_task_args->local_argsp: \n");
    lsci_mesh_task_args_t* mtargs = (lsci_mesh_task_args_t*)targs->local_argsp;
    printf("\trank: %lu\n", mtargs->rank);
    printf("\twidth: %lu\n", mtargs->global_width);
    printf("\theight: %lu\n", mtargs->global_height);
    printf("\tdepth: %lu\n", mtargs->global_depth);
    printf("\tlen: %lu\n", mtargs->sgb_len);
}
