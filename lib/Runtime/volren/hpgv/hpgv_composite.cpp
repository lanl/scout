/**
 * hpgv_composite.c
 *
 * Copyright (c) 2008 Hongfeng Yu
 *
 * Contact:
 * Hongfeng Yu
 * hfstudio@gmail.com
 * 
 *
 * All rights reserved.  May not be used, modified, or copied 
 * without permission.
 *
 */

#include "scout/Runtime/volren/hpgv/hpgv_composite_in.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace scout {

#define PARTIAL_IMAGE_MSG   100
#define FULL_IMAGE_MSG      101


#ifdef HPGV_TIMING
int MY_UPDATE_TIME          = HPGV_TIMING_UNIT_0;
int MY_COMPOSITE_TIME       = HPGV_TIMING_UNIT_1;
int MY_EXCHANGE_TIME        = HPGV_TIMING_UNIT_2;
int MY_BLEND_TIME           = HPGV_TIMING_UNIT_3;
int MY_EXG_BLEND_TIME       = HPGV_TIMING_UNIT_4;
int MY_GATHER_TIME          = HPGV_TIMING_UNIT_5;
int MY_SEND_BYTE            = HPGV_TIMING_UNIT_6;
int MY_RECV_BYTE            = HPGV_TIMING_UNIT_7;
int MY_BLEND_BYTE           = HPGV_TIMING_UNIT_8;
int MY_EXCHANGE_SPEED       = HPGV_TIMING_UNIT_9;
int MY_BLEND_SPEED          = HPGV_TIMING_UNIT_10;

int MY_STAGE_EXCHANGE_TIME  = HPGV_TIMING_UNIT_11;
int MY_STAGE_BLEND_TIME     = HPGV_TIMING_UNIT_31;
int MY_STAGE_SEND_BYTE      = HPGV_TIMING_UNIT_51;
int MY_STAGE_RECV_BYTE      = HPGV_TIMING_UNIT_71;
int MY_STAGE_SEND_MESSAGE   = HPGV_TIMING_UNIT_91;
int MY_STAGE_RECV_MESSAGE   = HPGV_TIMING_UNIT_101;
int MY_WORK_ID              = HPGV_TIMING_UNIT_121;

#endif

int MY_STAGE_MAX            = 20;
int MY_STAGE_COUNT          = 0;


#define COLOR_COMPOSITE(partialcolor, compositecolor) \
(*(compositecolor)     += (1-(*(compositecolor + 3)))*(*(partialcolor)), \
 *(compositecolor + 1) += (1-(*(compositecolor + 3)))*(*(partialcolor + 1)),\
 *(compositecolor + 2) += (1-(*(compositecolor + 3)))*(*(partialcolor + 2)), \
 *(compositecolor + 3) += (1-(*(compositecolor + 3)))*(*(partialcolor + 3)))




static swap_control_t *theSwapControl = NULL;

static int theSwapTimingSaveLocal = HPGV_FALSE;
static int theSwapTimingShowLocal = HPGV_FALSE;         
static int theSwapTimingSaveGlobal = HPGV_FALSE;
static int theSwapTimingShowGlobal = HPGV_FALSE;         

/**
 * hpgv_composite_disable
 *
 */
void
hpgv_composite_disable(int entry)
{
    switch(entry) {
    case HPGV_TIMING_SAVE_LOCAL : theSwapTimingSaveLocal = HPGV_FALSE;
        break;
    case HPGV_TIMING_SHOW_LOCAL : theSwapTimingShowLocal = HPGV_FALSE;
        break;
    case HPGV_TIMING_SAVE_GLOBAL : theSwapTimingSaveGlobal = HPGV_FALSE;
        break;
    case HPGV_TIMING_SHOW_GLOBAL : theSwapTimingShowGlobal = HPGV_FALSE;
        break;
    default: 
        fprintf(stderr, "Unsupported enumeration");    
    }
}

/**
 * hpgv_composite_enable
 *
 */
void
hpgv_composite_enable(int entry)
{
    switch(entry) {
    case HPGV_TIMING_SAVE_LOCAL : theSwapTimingSaveLocal = HPGV_TRUE;
        break;
    case HPGV_TIMING_SHOW_LOCAL : theSwapTimingShowLocal = HPGV_TRUE;
        break;
    case HPGV_TIMING_SAVE_GLOBAL : theSwapTimingSaveGlobal = HPGV_TRUE;
        break;
    case HPGV_TIMING_SHOW_GLOBAL : theSwapTimingShowGlobal = HPGV_TRUE;
        break;
    default: 
        fprintf(stderr, "Unsupported enumeration");    
    }
}

/**
 * swaptree_delete_treenode
 *
 */
static void 
swaptree_delete_treenode(int mpiid, swapnode_t *swapnode) 
{
    uint32_t i;

    if (swapnode == NULL)
        return;
    
    /* delete leaf */
    if (swapnode->childnum == 0) {
        if (swapnode->assignorder != NULL) {
            free(swapnode->assignorder);
        }
        free(swapnode);
        return;
    }
    
    /* delete children */
    for (i = 0; i < swapnode->childnum; i++) {
        swaptree_delete_treenode(mpiid, swapnode->children[i]);        
    }
    
    /* delete itself */
    if (swapnode->assignorder != NULL ) {
        free(swapnode->assignorder);
    }
    
    if (swapnode->children != NULL) {
        free(swapnode->children);
    }
    
    free(swapnode);
}


/**
 * swaptree_delete_tree
 *
 */
static void 
swaptree_delete_tree(int mpiid, swaptree_t *swaptree)
{
    swaptree_delete_treenode(mpiid, swaptree->root);
        
    free(swaptree);
}


/**
 * swaptree_get_split
 *
 */
static split_t *
swaptree_get_split(int mpiid, int workernum, split_t **wokersplit)
{
    split_t *split = wokersplit[workernum];
    
    HPGV_ASSERT_P(mpiid, split, "Can not find the split.", HPGV_ERROR);
    
    return split;
}

 
/**
 * swaptree_new_treenode
 *
 */
static void 
swaptree_new_treenode(int mpiid, split_t **wokersplit, 
                      swapnode_t *swapnode, uint16_t *maxlevel)
{   
    uint32_t workernum, i;          
    swapnode_t *child = NULL;
    split_t *childsplit;
 
    workernum = swapnode->workernum;
    
    if (workernum == 1) {
        swapnode->assignorder = (uint32_t *)calloc(swapnode->workernum, 
                                                    sizeof(uint32_t *));
        HPGV_ASSERT_P(mpiid, swapnode->assignorder, 
                    "Out of memory.", HPGV_ERR_MEM);                        
        swapnode->assignorder[0] = swapnode->firstworker;
        return;        
    }
    
    childsplit = swaptree_get_split(mpiid, workernum, wokersplit);
    
    swapnode->children = (swapnode_t **)calloc(childsplit->count, 
                          sizeof(swapnode_t *));
    HPGV_ASSERT_P(mpiid, swapnode->children, "Out of memory.", HPGV_ERR_MEM);    
    
    swapnode->childnum = childsplit->count;
        
    for (i = 0; i < swapnode->childnum; i++) {        
        /* allocate */
        child = (swapnode_t *)calloc(1, sizeof(swapnode_t));
        HPGV_ASSERT_P(mpiid, child, "Out of memory.", HPGV_ERR_MEM);
        
        child->workernum = childsplit->split[i];
        child->level = swapnode->level + 1;
        child->parent = swapnode;
        child->childnum = 0;
                
        if (i == 0) {
            child->firstworker = swapnode->firstworker;
        } else {
            child->firstworker = swapnode->children[i-1]->lastworker + 1;
        }
                
        child->lastworker = child->firstworker + child->workernum - 1;
        child->assignorder = NULL;
                
        if (*maxlevel < child->level) {
            *maxlevel = child->level;
        }
                
        if (child->workernum > 1) {
            swaptree_new_treenode(mpiid, wokersplit, 
                                  child, maxlevel);            
        } else {
                        
            child->assignorder = (uint32_t *)calloc(child->workernum, 
                                                sizeof(uint32_t *));
            HPGV_ASSERT_P(mpiid, child->assignorder, 
                        "Out of memory.", HPGV_ERR_MEM);
                        
            child->assignorder[0] = child->firstworker;
        }
        
        /* attach the child */
        swapnode->children[i] = child;
    }
}


/**
 * swapaux_array_compare
 *
 */
static int 
swapaux_array_compare(const void * a, const void * b)
{
    int aa = ((array_t*)(a))->count;
    int bb = ((array_t*)(b))->count;
    
    if (aa == bb) {
        return 0;
    }
    
    if (aa > bb) {
        return -1;
    }
    
    return 1;
}


/**
 * swaptree_order_assignment: bottom-up recursively order assignment
 *
 */
static void 
swaptree_order_assignment(uint32_t mpiid, swapnode_t *swapnode, 
                          uint16_t maxlevel)
{
    int i, count, alldone;    
    array_t *sorted = NULL;
    
    /* skip if swapnode is NULL */
    if (swapnode == NULL) {
        return;
    }
    
    
    if (swapnode->childnum == 0) {
        return;
        /*
        HPGV_ABORT_P(mpiid, 
                   "This is a no-leaf node. Child number can not be zero.",
                   HPGV_ERROR);
        */
    }
        
    /* generate assignment of children */
    for (i = 0; i < swapnode->childnum; i++) {
        if (swapnode->children[i]->assignorder == NULL) {
            /* order assignment for the non-leaf node */
            swaptree_order_assignment(mpiid, swapnode->children[i], maxlevel);
                                
        } else {            
            
            /* ===================================================== */
            /* BUG : for qkswap, a tree does not need to be comoplete */
            
            /* this is leaf node. verify if it is a complete tree */
            /*
            if (swapnode->children[i]->level != maxlevel) {
                HPGV_ABORT_P(mpiid, "It is not a complete tree", HPGV_ERROR);
            }
            */
            
            /* ===================================================== */
        }
    }
    
    /* allcoate the memory for assignment */
    swapnode->assignorder = (uint32_t *)calloc(swapnode->workernum, 
                             sizeof(uint32_t));
    HPGV_ASSERT_P(mpiid, swapnode->assignorder, "Out of memory", HPGV_ERR_MEM);
        
    sorted = (array_t *)calloc(swapnode->childnum, sizeof(array_t));
    HPGV_ASSERT_P(mpiid, sorted, "Out of memory", HPGV_ERR_MEM);
    
    for (i = 0; i < swapnode->childnum; i++) {
        sorted[i].count = swapnode->children[i]->workernum;
        sorted[i].cursor = 0;
        sorted[i].array = swapnode->children[i]->assignorder;
    }
    
    qsort(sorted, swapnode->childnum, sizeof(array_t), swapaux_array_compare);
        
    alldone = HPGV_TRUE;
    count = 0;            
    do{
        alldone = HPGV_TRUE;        
        for (i = 0; i < swapnode->childnum; i++) {
            if (sorted[i].cursor < sorted[i].count) {
                swapnode->assignorder[count++] 
                        = sorted[i].array[sorted[i].cursor];    
                sorted[i].cursor++;
                alldone = HPGV_FALSE;
            }                
        }    
    }while(alldone == HPGV_FALSE);    
    
    free(sorted);
}


/**
 * swaptree_new_tree
 *
 */
static swaptree_t * 
swaptree_new_tree(int mpiid, int workernum, split_t **wokersplit)
{
    swapnode_t *root;
    swaptree_t *swaptree;
    uint16_t maxlevel;
    
    /* root */
    root = (swapnode_t *)calloc(1, sizeof(swapnode_t));    
    HPGV_ASSERT_P(mpiid, root != NULL, "Out of memory.", HPGV_ERR_MEM);

    /* entry */
    root->workernum = workernum;
    root->level = 0;
    root->parent = NULL;
    root->childnum = 0;
    root->firstworker = 0;
    root->lastworker = workernum - 1;
    root->assignorder = NULL;
        
    maxlevel = 0;
        
    swaptree_new_treenode(mpiid, wokersplit, root, &maxlevel);
    swaptree_order_assignment(mpiid, root, maxlevel);
    
    /* swap tree */    
    swaptree = (swaptree_t *)calloc(1, sizeof(swaptree_t));
    HPGV_ASSERT_P(mpiid, swaptree, "Out of memory", HPGV_ERR_MEM);
    
    /* attach root to swap tree*/
    swaptree->root = root;
    swaptree->maxlevel = maxlevel;

    return swaptree;
}


/**
 * swaptree_set_schedule: generate schedule each processor each level
 *
 */
static void 
swaptree_set_schedule(swap_control_t *swapctrl, 
                      swapnode_t *swapnode,                     
                      swap_schedule_t **swapschedule)                            
{
    int mpiid = swapctrl->mpiid;
    int workid = swapctrl->workid;
    int *workid_2_mpiid = swapctrl->workid_2_mpiid;
    uint64_t usefulpixels = swapctrl->usefulpixels;
    uint16_t pixelsize = swapctrl->pixelsize;
    uint32_t maxstage = swapctrl->stagenum;    
    
    int32_t myworkid, mystage;
    uint32_t my_curstart, my_curend, my_prevstart, my_prevend;    
    uint32_t otherworkid, otherstart, otherend;
    uint32_t overlapstart, overlapend;
    uint32_t i, j;
    swap_message_t *newmsg;
    swap_schedule_t *schedule;
    swapnode_t *child = NULL;
        
    my_curstart = my_curend = my_prevstart = my_prevend = 0;
    
    /* skip leaf node */
    if (swapnode->childnum == 0) {
        return;
    }
        
    myworkid = -1;
    for (i = 0; i < swapnode->workernum; i++) {
        if (workid == swapnode->assignorder[i]) {
            myworkid = workid;
            mystage = maxstage - swapnode->level - 1;        
            my_curstart = BLOCK_LOW(i, swapnode->workernum, usefulpixels);
            my_curend = BLOCK_HIGH(i, swapnode->workernum, usefulpixels);
            break;
        }
    }
    
    if (myworkid == -1) {
        if (mpiid == swapctrl->root && !swapctrl->roothasimage) {
            return;
        }
        
        HPGV_ABORT_P(mpiid, "Can not find processor.", HPGV_ERROR);
    }
        
    /* generate a schedule for the current stage */
    schedule = (swap_schedule_t *)calloc(1, sizeof(swap_schedule_t));
    HPGV_ASSERT_P(mpiid, schedule, "Out of memory.", HPGV_ERR_MEM);    
            
    swapschedule[mystage] = schedule;
    schedule->mpiid = mpiid;
    schedule->curstage = mystage;
    
    schedule->bldoffset = my_curstart;
    schedule->bldcount = my_curend - my_curstart + 1;
    
    for (i = 0; i < swapnode->childnum; i++) {

        for (j = 0; j < swapnode->children[i]->workernum; j++) {
        
            otherworkid = swapnode->children[i]->assignorder[j];           
            otherstart = BLOCK_LOW(j, swapnode->children[i]->workernum, 
                                   usefulpixels);
            otherend = BLOCK_HIGH(j, swapnode->children[i]->workernum, 
                                  usefulpixels);

            if (otherworkid == myworkid) {
                my_prevstart = otherstart;
                my_prevend = otherend;
                
                HPGV_ASSERT_P(mpiid, child == NULL, 
                            "I can only be in one child node.", 
                            HPGV_ERR_MEM);
                                
                child = swapnode->children[i];
            }
            
            
            if ( !(my_curstart > otherend || my_curend < otherstart)) {

                overlapstart = my_curstart >= otherstart ?
                    my_curstart : otherstart;
                overlapend = my_curend <= otherend ?
                    my_curend : otherend;
                
                newmsg = (swap_message_t *)calloc(1, sizeof(swap_message_t));
                HPGV_ASSERT_P(mpiid, newmsg, "Out of memory,", HPGV_ERR_MEM);

                newmsg->mpiid = workid_2_mpiid[otherworkid];
                newmsg->offset = overlapstart;
                newmsg->recordcount = overlapend - overlapstart + 1;   
                newmsg->pixelsize = pixelsize;

                if (newmsg->mpiid >= swapctrl->totalprocnum) {
                    fprintf(stderr, "%d recv overflow. otherworkid %u; mpiid %u\n",
                            swapctrl->mpiid, otherworkid, newmsg->mpiid);
                }
                
                                
                if ( newmsg->mpiid != mpiid)
                {
                    newmsg->partial_image = 
                            (uint8_t *)calloc(newmsg->recordcount, pixelsize);
                    
                    HPGV_ASSERT_P(mpiid, newmsg->partial_image,
                                "Out of memory.", HPGV_ERR_MEM);

                    schedule->recv_count++;
                    schedule->recv_recordcount += newmsg->recordcount;
                }
                    
                if (schedule->first_recv == NULL) {
                    schedule->first_recv = newmsg;
                    schedule->last_recv = newmsg;
                } else {
                    schedule->last_recv->next = newmsg;
                    schedule->last_recv = newmsg;
                }
            }
        }
    }
        
    for (i = 0; i < swapnode->workernum; i++) {
        otherworkid = swapnode->assignorder[i];
        
        if (myworkid != otherworkid) {
            otherstart = BLOCK_LOW(i, swapnode->workernum, usefulpixels);
            otherend   = BLOCK_HIGH(i, swapnode->workernum, usefulpixels);
            
            if ( !(my_prevstart > otherend || my_prevend < otherstart)) {
                overlapstart = my_prevstart >= otherstart ?
                               my_prevstart : otherstart;
                overlapend = my_prevend <= otherend ?
                             my_prevend : otherend;
                
                newmsg = (swap_message_t *)calloc(1, sizeof(swap_message_t));
                HPGV_ASSERT_P(mpiid, newmsg, "Out of memory.", HPGV_ERR_MEM);

                newmsg->mpiid = workid_2_mpiid[otherworkid];
                newmsg->offset = overlapstart;
                newmsg->recordcount = overlapend - overlapstart + 1;
                newmsg->pixelsize = pixelsize;

                if (newmsg->mpiid >= swapctrl->totalprocnum) {
                    fprintf(stderr, "%d send overflow. otherworkid %u; mpiid %u\n",
                            swapctrl->mpiid, otherworkid, newmsg->mpiid);
                }

                if (newmsg->offset >= usefulpixels) {
                    fprintf(stderr, "%d overflow. offset %ld; total %ld\n",
                            swapctrl->mpiid,newmsg->offset, usefulpixels);
                }

                schedule->send_count++;
                schedule->send_recordcount += newmsg->recordcount;

                //newmsg->next = schedule->first_send;
                //schedule->first_send = newmsg;
                
                if (schedule->first_send == NULL) {
                    schedule->first_send = newmsg;
                    schedule->last_send = newmsg;
                } else {
                    schedule->last_send->next = newmsg;
                    schedule->last_send = newmsg;
                }
            }
        }
    }
    
    /* allocate structures for asynchronous MPI send/receive */
    schedule->isendreqs = 
        (MPI_Request *)calloc(schedule->send_count, sizeof(MPI_Request));
    HPGV_ASSERT_P(mpiid, schedule->isendreqs, "Out of memory.", HPGV_ERR_MEM);
    
    schedule->isendstats = 
        (MPI_Status *)calloc(schedule->send_count, sizeof(MPI_Status));
    HPGV_ASSERT_P(mpiid, schedule->isendstats, "Out of memory.", HPGV_ERR_MEM);

    
    schedule->irecvreqs = 
        (MPI_Request *)calloc(schedule->recv_count, sizeof(MPI_Request));
    HPGV_ASSERT_P(mpiid, schedule->irecvreqs, "Out of memory.", HPGV_ERR_MEM);
    
    schedule->irecvstats = 
        (MPI_Status *)calloc(schedule->recv_count, sizeof(MPI_Status));
    HPGV_ASSERT_P(mpiid, schedule->irecvstats, "Out of memory.", HPGV_ERR_MEM);
    
    HPGV_ASSERT_P(mpiid, child != NULL, "Can not find a child containing myself.",
                HPGV_ERROR);
    
    /* goto lower level */    
    swaptree_set_schedule(swapctrl, child, swapschedule);  
}


/**
 * swapaux_splitworker_drsend
 *
 */
static void
swapaux_splitworker_drsend(int mpiid, int workernum, split_t **split)
{   
    int count, i;
    split_t *newsplit = NULL;
        
    if (workernum == 0) {
        split[workernum] = NULL;
        return;
    }
    
    if (split[workernum] != NULL) {
        return;
    }
    
    newsplit = (split_t *)calloc(1, sizeof(split_t));
    HPGV_ASSERT_P(mpiid, newsplit, "Out of memory.", HPGV_ERR_MEM);
    
        
    if (workernum == 1) {        
        newsplit->count = 1;
        newsplit->split = (int *)calloc(1, sizeof(int));
        HPGV_ASSERT_P(mpiid, newsplit->split, "Out of memory.", HPGV_ERR_MEM);
        newsplit->split[0] = 1;
        split[workernum] = newsplit;
        return;
    }        
        
    count = workernum;
    newsplit->count = count;
    newsplit->split = (int *)calloc(count, sizeof(int));
    HPGV_ASSERT_P(mpiid, newsplit->split, "Out of memory.", HPGV_ERR_MEM);
    
    for (i = 0; i < count; i++) {
        newsplit->split[i] = 1;
    }
    
    split[workernum] = newsplit;
    
    /* recursively split */    
    swapaux_splitworker_drsend(mpiid, 1, split);
}



/**
 * swapaux_splitworker_ttswap
 *
 */
static void
swapaux_splitworker_ttswap(int mpiid, int workernum, split_t **split)
{
    int count, partition[3], lognum[3], i;
    split_t *newsplit = NULL;
    
    if (workernum == 0) {
        split[workernum] = NULL;
        return;
    }
    
    if (split[workernum] != NULL) {
        return;
    }
    
    newsplit = (split_t *)calloc(1, sizeof(split_t));
    HPGV_ASSERT_P(mpiid, newsplit, "Out of memory.", HPGV_ERR_MEM);        
    
    if (workernum == 1) {        
        newsplit->count = 1;
        newsplit->split = (int *)calloc(1, sizeof(int));
        HPGV_ASSERT_P(mpiid, newsplit->split, "Out of memory.", HPGV_ERR_MEM);
        newsplit->split[0] = 1;
        split[workernum] = newsplit;
        return;
    }        
    
    /* test if it can be divide into 2 */
    count = 2;
    partition[0] = workernum / 2;
    partition[1] = workernum - partition[0];

    lognum[0] = (int)log_2((double) (partition[0]));
    lognum[1] = (int)log_2((double) (partition[1]));
        
    /* if not then divide into 3 */
    if (lognum[0] != lognum[1]) {
        count = 3;
        partition[0] = partition[1] = workernum / 3;
        partition[2] = workernum - partition[0] - partition[1];
        
        /* verify 3 partition */
        for (i = 0; i < 3; i++) {
            lognum[i] = (int)log_2((double) (partition[i]));
        }
            
        if (lognum[0] != lognum[1] || lognum[0] != lognum[2]) {
            HPGV_ABORT_P(mpiid, "Can not partition a number", HPGV_ERROR);    
        }
    }    
    
    newsplit->count = count;
    newsplit->split = (int *)calloc(count, sizeof(int));
    HPGV_ASSERT_P(mpiid, newsplit->split, "Out of memory.", HPGV_ERR_MEM);
    
    for (i = 0; i < count; i++) {
        newsplit->split[i] = partition[i];
    }
    
    split[workernum] = newsplit;
    
    /* recursively split */
    for (i = 0; i < count; i++) {
        swapaux_splitworker_ttswap(mpiid, partition[i], split);
    }
}

/**
 * swap_update_wokersplit
 *
 */
static void
swap_update_wokersplit(swap_control_t *swapctrl)
{   
    int mpiid = swapctrl->mpiid;
    int workernum = swapctrl->workernum;
    
    swapctrl->workersplit = (split_t **)realloc(swapctrl->workersplit, 
                                        sizeof(split_t *) * (workernum + 1));
    memset(swapctrl->workersplit, 0, sizeof(split_t *) * (workernum + 1));
    HPGV_ASSERT_P(mpiid, swapctrl->workersplit, "Out of memory.", HPGV_ERR_MEM);
        
    switch(swapctrl->composite_type) {
        case HPGV_TTSWAP :     
            /* 2-3 swap */
            swapaux_splitworker_ttswap(mpiid, workernum, swapctrl->workersplit);
            break;
        case HPGV_DRSEND :
            /* direct send */
            swapaux_splitworker_drsend(mpiid, workernum, swapctrl->workersplit);
            break;        
        default:
            HPGV_ERR_MSG_P(mpiid, "Unsupported compositing type.");
    }
}


/**
 * swap_update_tree
 *
 */
static void
swap_update_tree(swap_control_t *swapctrl)
{
    int mpiid = swapctrl->mpiid;
    
    if (swapctrl->swaptree != NULL) {
        swaptree_delete_tree(mpiid, swapctrl->swaptree);
        swapctrl->swaptree = NULL;
    }
    
    swapctrl->swaptree = swaptree_new_tree(mpiid, swapctrl->workernum, 
                                           swapctrl->workersplit); 
}


/**
 * swapaux_depth_compare
 *
 */
static int 
swapaux_depth_compare(const void * a, const void * b)
{
    float aa = ((depth_t*)(a))->d;
    float bb = ((depth_t*)(b))->d;
    
    if (aa == bb) {
        return 0;
    }
    
    if (aa < bb) {
        return -1;
    }
    
    return 1;
}


/**
 * swap_update_order
 *
 */
static void
swap_update_order(swap_control_t *swapctrl)
{    
    int mpiid = swapctrl->mpiid;    
    int workernum = swapctrl->workernum;
    int totalprocnum = swapctrl->totalprocnum;
    int root = swapctrl->root;    
    MPI_Comm mpicomm = swapctrl->mpicomm;
    depth_t *dbuf = NULL;
    int i, count, workid;
    
    if (mpiid == root) {
                
        dbuf = (depth_t *)calloc(totalprocnum, sizeof(depth_t));
        HPGV_ASSERT_P(mpiid, dbuf, "Out of memory.", HPGV_ERR_MEM);
                
        MPI_Gather(&(swapctrl->depth), sizeof(depth_t), MPI_BYTE,
                    dbuf, sizeof(depth_t), MPI_BYTE, root, mpicomm);
        
        qsort(dbuf, totalprocnum, sizeof(depth_t), swapaux_depth_compare);
                
        swapctrl->workid_2_mpiid = (int *)realloc(swapctrl->workid_2_mpiid,
                                    workernum *sizeof(int));
        HPGV_ASSERT_P(mpiid, swapctrl->workid_2_mpiid, 
                    "Out of memory.", HPGV_ERR_MEM);
                
        count = 0;
        for (i = 0; i < totalprocnum; i++) {
            if (dbuf[i].mpiid == root && !swapctrl->roothasimage) {
                continue;
            }
            swapctrl->workid_2_mpiid[count++] = dbuf[i].mpiid;
        }
                
        MPI_Bcast(swapctrl->workid_2_mpiid, sizeof(int) * workernum, MPI_BYTE,
                  root, mpicomm);
        
        
        if (swapctrl->roothasimage == HPGV_TRUE) {
            workid = -1;
            for (i = 0; i < workernum; i++) {
                if (swapctrl->workid_2_mpiid[i] == mpiid) {
                    workid = i;
                    swapctrl->workid = i;
                    break;
                }
            }        
            HPGV_ASSERT_P(mpiid, workid != -1, 
                        "Can not find myself in the mapping table.", 
                        HPGV_ERROR);
        } else {
            swapctrl->workid = -1;
        }
        
        free(dbuf);
        
    } else {
        
        MPI_Gather(&(swapctrl->depth), sizeof(depth_t), MPI_BYTE,
                    NULL, sizeof(depth_t), MPI_BYTE, root, mpicomm);
        
        swapctrl->workid_2_mpiid = (int *)realloc(swapctrl->workid_2_mpiid,
                                    workernum *sizeof(int));
        HPGV_ASSERT_P(mpiid, swapctrl->workid_2_mpiid, 
                    "Out of memory.", HPGV_ERR_MEM);
        
        MPI_Bcast(swapctrl->workid_2_mpiid, sizeof(int) * workernum, MPI_BYTE,
                  root, mpicomm);
        
        workid = -1;
        for (i = 0; i < workernum; i++) {            
            if (swapctrl->workid_2_mpiid[i] == mpiid) {
                workid = i;
                swapctrl->workid = i;
                break;
            }
        }
        
        HPGV_ASSERT_P(mpiid, workid != -1, 
                    "Can not find myself in the mapping table.", HPGV_ERROR);

    }
}

/**
 * swap_update_worker
 *
 */
static void        
swap_update_worker(swap_control_t *swapctrl) 
{
    int mpiid = swapctrl->mpiid;    
    int *proc = NULL;
    int totalprocnum =swapctrl->totalprocnum;
    int i, count;
        
    
    proc = (int *)calloc(totalprocnum, sizeof(int));
    HPGV_ASSERT_P(mpiid, proc, "Out of memory.", HPGV_ERR_MEM);
    
    /* gather all mpi ids */
    MPI_Allgather(&(mpiid), sizeof(int), MPI_BYTE,
                  proc, sizeof(int), MPI_BYTE, swapctrl->mpicomm);
    
    swapctrl->workers = (int *)realloc(swapctrl->workers, 
                         swapctrl->workernum * sizeof(int));
    HPGV_ASSERT_P(mpiid, swapctrl->workers, "Out of memory.", HPGV_ERR_MEM);
    
    count = 0;
    for (i = 0; i < totalprocnum; i++) {
        /* skip root if root is not a worker */
        if (proc[i] == swapctrl->root && !swapctrl->roothasimage) {
            continue;
        }
        swapctrl->workers[count++] = proc[i];
    }
    
    free(proc);
}


/**
 * swap_delete_schedule: delete a schedule
 *
 */
static void 
swap_delete_schedule(swap_schedule_t **schedule, uint32_t stagenum)
{
    uint32_t i;
    swap_schedule_t *onesch;
    swap_message_t *onemsg;
    
    for (i = 0; i < stagenum; i++) {
        onesch = schedule[i];
        
        if (onesch == NULL) {
            continue;
        }
        
        while (onesch->first_send != NULL) {
            onemsg = onesch->first_send;
            onesch->first_send = onemsg->next;
            free(onemsg);            
        }
        
        while (onesch->first_recv != NULL) {
            onemsg = onesch->first_recv;
            onesch->first_recv = onemsg->next;
            if (onemsg != NULL && onemsg->partial_image !=  NULL) {
                free(onemsg->partial_image);
            }
            free(onemsg);
        }
        
        if (onesch->isendreqs != NULL) {
            free(onesch->isendreqs);
        }           
        
        if (onesch->isendstats != NULL) {
            free(onesch->isendstats);
        }
        
        if (onesch->irecvreqs != NULL) {
            free(onesch->irecvreqs);
        }           
        
        if (onesch->irecvstats != NULL) {
            free(onesch->irecvstats);
        }
        
        free(onesch);
    }    
    
    free(schedule);
}


/**
 * swap_update_schedule
 *
 */
static void
swap_update_schedule(swap_control_t *swapctrl)
{
    int mpiid = swapctrl->mpiid;
    int workid = -1;
    swapnode_t *root = NULL;
    int64_t i, start, end;
        
    if (swapctrl->schedule != NULL) {
        swap_delete_schedule(swapctrl->schedule, swapctrl->stagenum);
    }        
    
    swapctrl->stagenum = swapctrl->swaptree->maxlevel;
    swapctrl->schedule = (swap_schedule_t **)calloc(swapctrl->stagenum, 
                          sizeof(swap_schedule_t*));
    HPGV_ASSERT_P(mpiid, swapctrl->schedule, "Out of memory.", HPGV_ERR_MEM);
    
    swaptree_set_schedule(swapctrl, 
                          swapctrl->swaptree->root, 
                          swapctrl->schedule);
    
    root = swapctrl->swaptree->root;
        
    swapctrl->finalassign = (int64_t *)realloc(swapctrl->finalassign,
                             sizeof(int64_t) * root->workernum * 2);
    HPGV_ASSERT_P(mpiid, swapctrl->finalassign, "Out of memory.", HPGV_ERR_MEM);
    
    for (i = 0; i < root->workernum; i++) {
        workid = root->assignorder[i];
        
        start = BLOCK_LOW(i, root->workernum, swapctrl->usefulpixels);
        end   = BLOCK_HIGH(i, root->workernum, swapctrl->usefulpixels);
        
        swapctrl->finalassign[workid * 2] = start;
        swapctrl->finalassign[workid * 2 + 1] = end;
    }
    
}


/**
 * swap_control_exchange
 *
 */
static void
swap_control_exchange(swap_control_t *swapctrl, 
                      swap_schedule_t *schedule,
                      int stage)
{
    int mpiid = swapctrl->mpiid;
    MPI_Comm mpicomm = swapctrl->mpicomm;
    int curstage = schedule->curstage;
    
    uint32_t isendnum, irecvnum;
    uint32_t bytesize, byteoffset;
    swap_message_t *inmessage, *outmessage;
    hpgv_pixel_t *sendbuf = NULL;
    
    /* Post async receive */
    irecvnum = 0;
    inmessage = schedule->first_recv;
    while (inmessage != NULL) {
        
        if (inmessage->mpiid != mpiid) {
            
            bytesize = inmessage->recordcount * inmessage->pixelsize;

            /* recv */
            MPI_Irecv(inmessage->partial_image, bytesize, MPI_BYTE,
                      inmessage->mpiid, PARTIAL_IMAGE_MSG + curstage, mpicomm,
                      &(schedule->irecvreqs[irecvnum]));
            irecvnum++;

            /* ========= timing ===========*/
            HPGV_TIMING_INCREASE(MY_RECV_BYTE, bytesize);
            HPGV_TIMING_INCREASE(MY_STAGE_RECV_BYTE + stage, bytesize);
            HPGV_TIMING_INCREASE(MY_STAGE_RECV_MESSAGE + stage, 1);
        }

        inmessage = inmessage->next;
    }
    
    /* isend all the recordcount information to the destination */
    isendnum = 0;
    outmessage = schedule->first_send;
    while (outmessage != NULL) {
        
        uint8_t *partial_image = swapctrl->partial_image[curstage % 2];
        uint8_t *ptr = NULL;

        bytesize = outmessage->recordcount * outmessage->pixelsize;
        byteoffset = outmessage->offset * outmessage->pixelsize;
        ptr = &(partial_image[byteoffset]);

        /* send */
        MPI_Isend(ptr, bytesize, MPI_BYTE, outmessage->mpiid,
                  PARTIAL_IMAGE_MSG + curstage, mpicomm,
                  &(schedule->isendreqs[isendnum]));
        isendnum++;

        /* ========= timing ===========*/
        HPGV_TIMING_INCREASE(MY_SEND_BYTE, bytesize);
        HPGV_TIMING_INCREASE(MY_STAGE_SEND_BYTE + stage, bytesize);
        HPGV_TIMING_INCREASE(MY_STAGE_SEND_MESSAGE + stage, 1);

        outmessage = outmessage->next;
    }
    
    if (irecvnum != 0) {
        MPI_Waitall(irecvnum, schedule->irecvreqs, schedule->irecvstats);
    }
    
    if (isendnum != 0) {
        MPI_Waitall(isendnum, schedule->isendreqs, schedule->isendstats);
    }
    
    if (sendbuf != NULL) {
        free(sendbuf);
    }
}



/**
 * swap_control_blend_float
 *
 */
static void
swap_control_blend_float(swap_control_t *swapctrl, swap_schedule_t *schedule)
{
    swap_message_t *inmessage;
    uint64_t offset, recordcount, i, id;
    int mpiid = swapctrl->mpiid;
    uint32_t pixelsize = swapctrl->pixelsize;
    int curstage = schedule->curstage;

    float *outimage = (float *)(swapctrl->partial_image[1 - curstage % 2]);
    
    float *myimage = (float *)(swapctrl->partial_image[curstage % 2]);
    
    float *inimage = NULL;

    float *compositecolor = NULL;

    float *partialcolor = NULL;

    float alphafactor = 0;

    int formatsize = hpgv_formatsize(HPGV_RGBA);
    
    inmessage = schedule->first_recv;

    memset(&(outimage[schedule->bldoffset * formatsize]), 0,
           schedule->bldcount * pixelsize);

    while (inmessage != NULL) {
        
        offset = inmessage->offset;
        recordcount = inmessage->recordcount;
        
        /* ========= timing ===========*/
        HPGV_TIMING_INCREASE(MY_BLEND_BYTE, 
                                recordcount * pixelsize);
        
        if (inmessage->mpiid != mpiid) {
            inimage = (float *)(inmessage->partial_image);
        } else {
            inimage = &(myimage[offset * formatsize]);
        }

        compositecolor = &(outimage[offset * formatsize]);
        partialcolor = inimage;

        id = 0;
        for (i = 0; i < recordcount; i++) {
            
            alphafactor = 1.0 - compositecolor[id+3];
            compositecolor[id]   += (alphafactor)*(partialcolor[id]);
            compositecolor[id+1] += (alphafactor)*(partialcolor[id+1]);
            compositecolor[id+2] += (alphafactor)*(partialcolor[id+2]);
            compositecolor[id+3] += (alphafactor)*(partialcolor[id+3]);
            
            id += formatsize;

            /*
            COLOR_COMPOSITE(&(inimage[i*4]),
            &(outimage[(offset + i)*4]));
            */
        }
        
        inmessage = inmessage->next;
    }
}

/**
 * swap_control_blend_uint16
 *
 */
static void
swap_control_blend_uint16(swap_control_t *swapctrl, swap_schedule_t *schedule)
{
    swap_message_t *inmessage;
    uint64_t offset, recordcount, i, id;
    int mpiid = swapctrl->mpiid;
    uint32_t pixelsize = swapctrl->pixelsize;
    int curstage = schedule->curstage;
    
    uint16_t *outimage = (uint16_t *)(swapctrl->partial_image[1 - curstage % 2]);
    
    uint16_t *myimage = (uint16_t *)(swapctrl->partial_image[curstage % 2]);
    
    uint16_t *inimage = NULL;
    
    uint16_t *compositecolor = NULL;
    
    uint16_t *partialcolor = NULL;
    
    float alphafactor = 0;
    
    int formatsize = hpgv_formatsize(HPGV_RGBA);
    
    inmessage = schedule->first_recv;
    
    memset(&(outimage[schedule->bldoffset * formatsize]), 0,
           schedule->bldcount * pixelsize);

    float r, g, b, a;
    
    while (inmessage != NULL) {
        
        offset = inmessage->offset;
        recordcount = inmessage->recordcount;
        
        /* ========= timing ===========*/
        HPGV_TIMING_INCREASE(MY_BLEND_BYTE,
                             recordcount * pixelsize);
        
        if (inmessage->mpiid != mpiid) {
            inimage = (uint16_t *)(inmessage->partial_image);
        } else {
            inimage = &(myimage[offset * formatsize]);
        }
        
        compositecolor = &(outimage[offset * formatsize]);
        partialcolor = inimage;
        
        id = 0;
        for (i = 0; i < recordcount; i++) {
            
            alphafactor = 1.0 - compositecolor[id+3] / 65535.0f;

            r = compositecolor[id]   + (alphafactor)*(partialcolor[id]);
            g = compositecolor[id+1] + (alphafactor)*(partialcolor[id+1]);
            b = compositecolor[id+2] + (alphafactor)*(partialcolor[id+2]);
            a = compositecolor[id+3] + (alphafactor)*(partialcolor[id+3]);
            
            compositecolor[id]   = r < 0xFFFF ? (uint16_t)(r) : 0xFFFF;
            compositecolor[id+1] = g < 0xFFFF ? (uint16_t)(g) : 0xFFFF;
            compositecolor[id+2] = b < 0xFFFF ? (uint16_t)(b) : 0xFFFF;
            compositecolor[id+3] = a < 0xFFFF ? (uint16_t)(a) : 0xFFFF;
           

            id += formatsize;
            
            /*
            COLOR_COMPOSITE(&(inimage[i*4]),
            &(outimage[(offset + i)*4]));
            */
        }
        
        inmessage = inmessage->next;
    }
}


/**
 * swap_control_blend_uint8
 *
 */
static void
swap_control_blend_uint8(swap_control_t *swapctrl, swap_schedule_t *schedule)
{
    swap_message_t *inmessage;
    uint64_t offset, recordcount, i, id;
    int mpiid = swapctrl->mpiid;
    uint32_t pixelsize = swapctrl->pixelsize;
    int curstage = schedule->curstage;
    
    uint8_t *outimage = (uint8_t *)(swapctrl->partial_image[1 - curstage % 2]);
    
    uint8_t *myimage = (uint8_t *)(swapctrl->partial_image[curstage % 2]);
    
    uint8_t *inimage = NULL;
    
    uint8_t *compositecolor = NULL;
    
    uint8_t *partialcolor = NULL;
    
    float alphafactor = 0;
    
    int formatsize = hpgv_formatsize(HPGV_RGBA);
    
    inmessage = schedule->first_recv;
    
    memset(&(outimage[schedule->bldoffset * formatsize]), 0,
           schedule->bldcount * pixelsize);

    float r, g, b, a;
    
    while (inmessage != NULL) {
        
        offset = inmessage->offset;
        recordcount = inmessage->recordcount;
        
        /* ========= timing ===========*/
        HPGV_TIMING_INCREASE(MY_BLEND_BYTE,
                             recordcount * pixelsize);
        
        if (inmessage->mpiid != mpiid) {
            inimage = (uint8_t *)(inmessage->partial_image);
        } else {
            inimage = &(myimage[offset * formatsize]);
        }
        
        compositecolor = &(outimage[offset * formatsize]);
        partialcolor = inimage;
        
        id = 0;
        for (i = 0; i < recordcount; i++) {
            
            alphafactor = 1.0 - compositecolor[id+3] / 255.0f;

            r = compositecolor[id]   + (alphafactor)*(partialcolor[id]);
            g = compositecolor[id+1] + (alphafactor)*(partialcolor[id+1]);
            b = compositecolor[id+2] + (alphafactor)*(partialcolor[id+2]);
            a = compositecolor[id+3] + (alphafactor)*(partialcolor[id+3]);

            compositecolor[id]   = r < 0xFF ? (uint8_t)(r) : 0xFF;
            compositecolor[id+1] = g < 0xFF ? (uint8_t)(g) : 0xFF;
            compositecolor[id+2] = b < 0xFF ? (uint8_t)(b) : 0xFF;
            compositecolor[id+3] = a < 0xFF ? (uint8_t)(a) : 0xFF;

            id += formatsize;
            
            /*
            COLOR_COMPOSITE(&(inimage[i*4]),
            &(outimage[(offset + i)*4]));
            */
        }
        
        inmessage = inmessage->next;
    }
}

        
/**
 * swap_control_gather
 *
 */
static void
swap_control_gather(swap_control_t *swapctrl)
{
    uint32_t mpiid = swapctrl->mpiid;
    uint32_t workid, sendid;
    uint32_t workernum = swapctrl->workernum;
    uint64_t bytesize, recordcount;
    int64_t start, end;
    uint8_t *partial_image = NULL;  
    uint8_t *final_image = NULL;  
    MPI_Status status;
    uint16_t pixelsize = swapctrl->pixelsize;    
    int curstage = swapctrl->stagenum + 1;
    
    /* I collect all the partial images from the other processors */
    if (mpiid == swapctrl->root) {        
                
        for (workid = 0; workid < workernum; workid++) {
            
            start = swapctrl->finalassign[workid * 2];
            
            if (start == -1) {
                continue;
            }            
            end = swapctrl->finalassign[workid * 2 + 1];            
            
            recordcount = end - start + 1;                                
                  
            bytesize = recordcount * pixelsize;            
            
            final_image = &(swapctrl->final_image[start * pixelsize]);
            
            if (swapctrl->roothasimage && workid == swapctrl->workid) {
                /* copy mine */                  
                memcpy(final_image,
                       &(swapctrl->partial_image[1-curstage%2][start * pixelsize]),
                       recordcount * pixelsize);       
                
            } else {         
                sendid = swapctrl->workid_2_mpiid[workid];
                
                /* recv from others */ 
                MPI_Recv(final_image, bytesize, MPI_CHAR, sendid,
                         FULL_IMAGE_MSG, swapctrl->mpicomm, &status);
                
            }
        }
        
    } else {
        
        workid = swapctrl->workid;
        
        /* I send my partial image to processor 0 */
        start = swapctrl->finalassign[workid * 2];
        
        if (start == -1) {
            return;
        }
            
        end = swapctrl->finalassign[workid * 2 + 1];
        
        
        recordcount = end - start + 1;                                      
        
        bytesize = recordcount * pixelsize;
        
        partial_image
            = &(swapctrl->partial_image[1-curstage%2][start * pixelsize]);
        
        MPI_Send(partial_image, bytesize, MPI_CHAR, swapctrl->root, 
                 FULL_IMAGE_MSG, swapctrl->mpicomm);
    }
        
}


/**
 * swap_control_composite
 *
 */
static void
swap_control_composite(swap_control_t *swapctrl)
{
    
    uint32_t stage;

    /* if there is no worker, we don't need image compositing */
    if (swapctrl->workernum < 1) {
        return;
    }
    
    /* ========= timing ===========*/
    HPGV_TIMING_BARRIER(swapctrl->mpicomm);
    HPGV_TIMING_BEGIN(MY_COMPOSITE_TIME);
    HPGV_TIMING_BEGIN(MY_EXG_BLEND_TIME);
    
    HPGV_TIMING_COUNT(MY_COMPOSITE_TIME);
    HPGV_TIMING_COUNT(MY_EXCHANGE_TIME);
    HPGV_TIMING_COUNT(MY_BLEND_TIME);
    HPGV_TIMING_COUNT(MY_EXG_BLEND_TIME);
    HPGV_TIMING_COUNT(MY_SEND_BYTE);
    HPGV_TIMING_COUNT(MY_RECV_BYTE);
    HPGV_TIMING_COUNT(MY_BLEND_BYTE);

    MY_STAGE_COUNT = swapctrl->stagenum;
    HPGV_TIMING_INCREASE(MY_WORK_ID, swapctrl->workid);
    HPGV_TIMING_COUNT(MY_WORK_ID);
    
    /* do image compositing */
    for (stage = 0; stage < swapctrl->stagenum; stage++) {        

        HPGV_ASSERT_P(swapctrl->mpiid, stage < MY_STAGE_MAX,
                    "Exceed the maximum stage number.", HPGV_ERROR);

        if (swapctrl->schedule[stage] == NULL) {
            continue;
        }
                
        /* ========= timing =========== */
        HPGV_TIMING_BEGIN(MY_EXCHANGE_TIME);
        HPGV_TIMING_BEGIN(MY_STAGE_EXCHANGE_TIME + stage);
        HPGV_TIMING_COUNT(MY_STAGE_EXCHANGE_TIME + stage);
        HPGV_TIMING_COUNT(MY_STAGE_SEND_BYTE + stage);
        HPGV_TIMING_COUNT(MY_STAGE_RECV_BYTE + stage);
        HPGV_TIMING_COUNT(MY_STAGE_SEND_MESSAGE + stage);
        HPGV_TIMING_COUNT(MY_STAGE_RECV_MESSAGE + stage);
        
        /* exchange */
        swap_control_exchange(swapctrl, swapctrl->schedule[stage], stage);


        /* ========= timing ===========*/
        HPGV_TIMING_END(MY_EXCHANGE_TIME);
        HPGV_TIMING_END(MY_STAGE_EXCHANGE_TIME + stage);
        
        /* ========= timing ===========*/
        HPGV_TIMING_BEGIN(MY_BLEND_TIME);
        HPGV_TIMING_BEGIN(MY_STAGE_BLEND_TIME + stage);
        HPGV_TIMING_COUNT(MY_STAGE_BLEND_TIME + stage);
        
        /* blend */
        switch(swapctrl->imagetype) {
        case HPGV_FLOAT:
            swap_control_blend_float(swapctrl, swapctrl->schedule[stage]);
            break;
        case HPGV_UNSIGNED_SHORT:
            swap_control_blend_uint16(swapctrl, swapctrl->schedule[stage]);
            break;
        case HPGV_UNSIGNED_BYTE:
            swap_control_blend_uint8(swapctrl, swapctrl->schedule[stage]);
            break;
        default:
            HPGV_ABORT_P(swapctrl->mpiid, "Unsupported pixel format.", HPGV_ERROR);
        }
        
        /* ========= timing ===========*/
        HPGV_TIMING_END(MY_BLEND_TIME);
        HPGV_TIMING_END(MY_STAGE_BLEND_TIME + stage);
    }
    
    /* ========= timing ===========*/
    HPGV_TIMING_END(MY_EXG_BLEND_TIME);
    
    /* ========= timing ===========*/
    HPGV_TIMING_BARRIER(swapctrl->mpicomm);
    HPGV_TIMING_BEGIN(MY_GATHER_TIME);
    HPGV_TIMING_COUNT(MY_GATHER_TIME);

    swap_control_gather(swapctrl);

    /* ========= timing ===========*/
    HPGV_TIMING_END(MY_GATHER_TIME);
    
    /* ========= timing ===========*/
    HPGV_TIMING_END(MY_COMPOSITE_TIME);
    
    /* ========= timing ===========*/
#ifdef HPGV_TIMING
    if (HPGV_TIMING_GET(MY_EXCHANGE_TIME) != 0) {
        double exgspeed 
            = (HPGV_TIMING_GET(MY_SEND_BYTE) + HPGV_TIMING_GET(MY_RECV_BYTE)) 
            / HPGV_TIMING_GET(MY_EXCHANGE_TIME);
        
        exgspeed /= 1e6;
    
        HPGV_TIMING_INCREASE(MY_EXCHANGE_SPEED, exgspeed);
        HPGV_TIMING_COUNT(MY_EXCHANGE_SPEED);
    }
    
    /* ========= timing ===========*/    
    if (HPGV_TIMING_GET(MY_BLEND_TIME) != 0) {
        double blendspeed 
            = HPGV_TIMING_GET(MY_BLEND_BYTE) / HPGV_TIMING_GET(MY_BLEND_TIME);
        
        blendspeed /= 1e6;
        
        HPGV_TIMING_INCREASE(MY_BLEND_SPEED, blendspeed);                        
        HPGV_TIMING_COUNT(MY_BLEND_SPEED);
    }
#endif
}


/**
 * swap_control_update
 *
 */
static int
swap_control_update(swap_control_t *swapctrl, 
                    int width, int height, 
                    int format, int type,
                    void *partialpixels,
                    void *finalpixels,
                    float depth, 
                    int root, 
                    MPI_Comm mpicomm,
                    composite_t composite_type) 
{   
    int mpiid = 0;
    
    int workernum = 0;
    int t_workernum = 0;
    int totalprocnum = 0;
    int roothasimage = HPGV_FALSE; 
    int pixelsize = 0;
    int i;
    
    update_t lu, gu;
    memset(&lu, 0, sizeof(update_t));
    memset(&gu, 0, sizeof(update_t));
        
    /* my rank */
    MPI_Comm_rank(mpicomm, &mpiid);    
    
    /* pixel size */
    pixelsize = hpgv_formatsize(format) * hpgv_typesize(type);
    
    /* total compositing processor number, including root */
    workernum = 1;
    totalprocnum = 0;
    t_workernum = 0;
    MPI_Allreduce(&workernum, &totalprocnum, 1,                  
                  MPI_INT, MPI_SUM, mpicomm);
    
    /* check if root has image */
    roothasimage = HPGV_FALSE;
    if (mpiid == root && partialpixels != NULL && width != 0 && height != 0) {
        roothasimage = HPGV_TRUE;
    }
    MPI_Bcast(&roothasimage, sizeof(int), MPI_BYTE, root, mpicomm);
    
    if (roothasimage == HPGV_TRUE) {
        t_workernum = totalprocnum;
    } else {
        t_workernum = totalprocnum - 1;
    }
    
    /* update everything if workernum, roothasimage, mpicomm, workernum
       or mpiid changed */
    if (swapctrl->workernum != t_workernum || 
        swapctrl->totalprocnum != totalprocnum ||
        swapctrl->roothasimage != roothasimage ||
        swapctrl->mpicomm != mpicomm ||        
        swapctrl->mpiid != mpiid ||
        swapctrl->root != root ||
        swapctrl->composite_type != composite_type) 
    {
        swapctrl->workernum = t_workernum;
        swapctrl->totalprocnum = totalprocnum;
        swapctrl->roothasimage = roothasimage;
        swapctrl->mpicomm = mpicomm;        
        swapctrl->mpiid = mpiid;
        swapctrl->root = root;
        swapctrl->composite_type = composite_type;
        
        lu.worker = HPGV_TRUE;
        lu.workersplit = HPGV_TRUE;
        lu.tree = HPGV_TRUE;
        lu.order = HPGV_TRUE;
        lu.schedule = HPGV_TRUE;        
    }

    /* check if I need to update compositing tree and schedule */            
    if (swapctrl->image_size_x != width || 
        swapctrl->image_size_y != height||
        swapctrl->imageformat != format ||
        swapctrl->imagetype != type)        
    {        
        swapctrl->image_size_x = width;
        swapctrl->image_size_y = height;        
        swapctrl->totalpixels = swapctrl->image_size_x * swapctrl->image_size_y;
        swapctrl->usefulpixels = swapctrl->totalpixels; 
        swapctrl->imageformat = format;
        swapctrl->imagetype = type;
        swapctrl->pixelsize = pixelsize;                
        swapctrl->totalpixelbytes = swapctrl->totalpixels * swapctrl->pixelsize; 
        
        if (!(mpiid == root && roothasimage == HPGV_FALSE)) {
            
            HPGV_ASSERT_P(mpiid, partialpixels, 
                          "There are no input pixels.", HPGV_ERR_MEM);

            if (swapctrl->imageformat == HPGV_PIXEL) {
                
                swapctrl->partial_image[0] 
                        = (uint8_t *)realloc(swapctrl->partial_image[0],
                                             swapctrl->totalpixelbytes);
                
                swapctrl->partial_image[1]
                        = (uint8_t *)realloc(swapctrl->partial_image[1],
                                             swapctrl->totalpixelbytes);
    
                swapctrl->collect_image
                        = (uint8_t *)realloc(swapctrl->collect_image,
                                             swapctrl->totalpixelbytes);
                                
                HPGV_ASSERT_P(mpiid,
                            swapctrl->partial_image[0] &&
                            swapctrl->partial_image[1] &&
                            swapctrl->collect_image,        
                            "There are no input pixels.", HPGV_ERR_MEM);
                
            } else {

                swapctrl->partial_image[0]
                    = (uint8_t *)realloc(swapctrl->partial_image[0], 
                                         swapctrl->totalpixelbytes);
    
                swapctrl->partial_image[1]
                    = (uint8_t *)realloc(swapctrl->partial_image[1],
                                         swapctrl->totalpixelbytes);
    
                HPGV_ASSERT_P(mpiid,
                            swapctrl->partial_image[0] &&
                            swapctrl->partial_image[1],
                            "There are no input pixels.", HPGV_ERR_MEM);
            }
        }
        
        lu.schedule = HPGV_TRUE;
    }

     /* clear the buffer */
    if (!(mpiid == root && roothasimage == HPGV_FALSE)) {
        if (swapctrl->imageformat == HPGV_PIXEL) {         
            
            memcpy(swapctrl->partial_image[0], partialpixels, 
                   swapctrl->totalpixelbytes);
            memset(swapctrl->partial_image[1], 0, swapctrl->totalpixelbytes);
            memset(swapctrl->collect_image, 0, swapctrl->totalpixelbytes);
            
            hpgv_pixel_t *partial_image 
                = (hpgv_pixel_t *)(swapctrl->partial_image[0]);                
            
            hpgv_pixel_t *collect_image 
                = (hpgv_pixel_t *)(swapctrl->collect_image);
                        
            for (i = 0; i < swapctrl->totalpixels; i++) {
                if (GET_FACE(partial_image[i].flags) == FACE_BOTH) {
                    collect_image[i] = partial_image[i];
                    SET_USELESS(partial_image[i].flags); 
                }
            }                     
            
        } else {
            memcpy(swapctrl->partial_image[0], partialpixels, 
                   swapctrl->totalpixelbytes);
            memset(swapctrl->partial_image[1], 0, swapctrl->totalpixelbytes);
        }
    }
    if (mpiid == root) {
        
        HPGV_ASSERT_P(mpiid, finalpixels, 
                    "The output buffer is empty.", HPGV_ERR_MEM);
        
        swapctrl->final_image = (uint8_t *)finalpixels;
        
        memset(swapctrl->final_image, 0, swapctrl->totalpixelbytes);
    }    
    
    /* check if I need to update ordering and schedule */
    if (swapctrl->depth.d != depth || !swapctrl->depth.init) {        
        swapctrl->depth.d = depth;
        swapctrl->depth.mpiid = mpiid;        
        lu.order = HPGV_TRUE;
        lu.schedule = HPGV_TRUE;
        swapctrl->depth.init = HPGV_TRUE; 
    }
    
    /* do nothing if there is no worker */    
    if (swapctrl->workernum == 0) {
        return HPGV_TRUE;
    }
    
    /* check if any one needs update */
    MPI_Allreduce(&lu, &gu, sizeof(update_t ), MPI_BYTE, MPI_BOR, mpicomm);
    
    /* return if we don't need to upate anything */
    if (!gu.worker && !gu.order && !gu.workersplit && 
        !gu.tree && !gu.schedule)
    {
        return HPGV_TRUE;
    }    
    
    
    /* ========= timing ===========*/
    HPGV_TIMING_BARRIER(mpicomm);
    HPGV_TIMING_BEGIN(MY_UPDATE_TIME);
    HPGV_TIMING_COUNT(MY_UPDATE_TIME);
    
    /* we need to update worker */
    if (gu.worker == HPGV_TRUE) {
        swap_update_worker(swapctrl);
    }    
    
    /* we need to update ordering */
    if (gu.order == HPGV_TRUE) {
        swap_update_order(swapctrl);
    }
    
    /* we need to update cpu partition */
    if (gu.workersplit == HPGV_TRUE) {        
        swap_update_wokersplit(swapctrl);
    }
    
    /* we need to update tree */
    if (gu.tree == HPGV_TRUE) {
        swap_update_tree(swapctrl);
    }
    
    /* we need to update the schedule */
    if (gu.schedule == HPGV_TRUE) {        
        swap_update_schedule(swapctrl);
    }
        
    /* ========= timing ===========*/
    HPGV_TIMING_END(MY_UPDATE_TIME);
    
    return HPGV_TRUE;
}


/**
 * hpgv_composite_init
 *
 */
int
hpgv_composite_init(MPI_Comm mpicomm)
{
    int mpiid = 0;
    MPI_Comm_rank(mpicomm, &mpiid);
    
    if( theSwapControl == NULL) {
        theSwapControl = (swap_control_t *)calloc(1, sizeof(swap_control_t));
    }    
    
    if (theSwapControl == NULL) {
        HPGV_ERR_MSG_P(mpiid, "Can not initialize compositing modelue.");
        return HPGV_FALSE;
    }
    
    
    return HPGV_TRUE;
}


/**
 * hpgv_composite
 *
 */
int
hpgv_composite_finalize(MPI_Comm mpicomm)
{
    int mpiid = 0;
    int i;
    MPI_Comm_rank(mpicomm, &mpiid);
    
    swap_control_t *swapctrl = theSwapControl;    
    
    if (swapctrl == NULL) {
        return HPGV_TRUE;
    }


    /* free memory */
    if (swapctrl->workid_2_mpiid != NULL) {
        free(swapctrl->workid_2_mpiid);
    }
    
    if (swapctrl->workers != NULL) {
        free(swapctrl->workers);
    }
    
    if (swapctrl->swaptree != NULL) {
        swaptree_delete_tree(mpiid, swapctrl->swaptree);
    }
    
    if (swapctrl->schedule != NULL) {
        swap_delete_schedule(swapctrl->schedule, swapctrl->stagenum);
    }
    
    if (swapctrl->partial_image[0] != NULL) 
    {
        free(swapctrl->partial_image[0]);
    }
        
    if (swapctrl->partial_image[1] != NULL) {
        free(swapctrl->partial_image[1]);
    }
    
    if (swapctrl->collect_image != NULL) {
        free(swapctrl->collect_image);
    }
    
    
    if (swapctrl->finalassign != NULL) {
        free(swapctrl->finalassign);
    }
    
    if (swapctrl->workersplit != NULL) {
        for (i = 0; i < swapctrl->workernum + 1; i++) {
            if (swapctrl->workersplit[i] != NULL) {
                if (swapctrl->workersplit[i]->split != NULL) {
                    free(swapctrl->workersplit[i]->split);
                }
                free(swapctrl->workersplit[i]);
            }
        } 
        
        free(swapctrl->workersplit);
    }
    
    free(swapctrl);
    theSwapControl = NULL;
    
    return HPGV_TRUE;
}


/**
 * hpgv_composite_valid
 *
 */
int
hpgv_composite_valid()
{
    if (theSwapControl == NULL) {
        return HPGV_FALSE;
    }
    
    return HPGV_TRUE;
}


/**
 * hpgv_composite
 *
 */
int  
hpgv_composite(int width, int height, int format, int type, 
               void *partialpixels, void *finalpixels, float depth,
               int root, MPI_Comm mpicomm, composite_t composite_type)
{
    
    char context[MAXLINE];
    int mpiid = 0;
    MPI_Comm_rank(mpicomm, &mpiid);
    
    //hpgv_msg_p(mpiid, root, "in hpgv_composite\n");
    if (theSwapControl == NULL) {
        hpgv_msg_p(mpiid, root, 
                   "Compositing modelue has not been intialized. Quit compositing.");
        return HPGV_FALSE;
    }
        
    if (width == 0 || height == 0) {
        return HPGV_TRUE;
    }
    
    /* ========= timing ===========*/
    /*  initalize timing module   */
    if (!HPGV_TIMING_VALID()) {
        HPGV_TIMING_INIT(root, mpicomm);
    }

    static int init_timing = HPGV_FALSE;

    if (init_timing == HPGV_FALSE) {
        init_timing = HPGV_TRUE;
        
        HPGV_TIMING_NAME(MY_UPDATE_TIME, "T_update");
        HPGV_TIMING_NAME(MY_COMPOSITE_TIME, "T_comp");
        HPGV_TIMING_NAME(MY_EXCHANGE_TIME, "T_exg");
        HPGV_TIMING_NAME(MY_BLEND_TIME, "T_blend");
        HPGV_TIMING_NAME(MY_EXG_BLEND_TIME, "T_exgbld");
        HPGV_TIMING_NAME(MY_GATHER_TIME, "T_gather");
        HPGV_TIMING_NAME(MY_SEND_BYTE, "C_send");
        HPGV_TIMING_NAME(MY_RECV_BYTE, "C_recv");
        HPGV_TIMING_NAME(MY_BLEND_BYTE, "C_blend");
        HPGV_TIMING_NAME(MY_EXCHANGE_SPEED, "B_exg");
        HPGV_TIMING_NAME(MY_BLEND_SPEED, "B_blend");
        
        if (theSwapTimingSaveLocal) {
            HPGV_TIMING_SAVELOCAL(HPGV_TRUE);
        }
        
        if (theSwapTimingShowLocal) {
            HPGV_TIMING_SHOWLOCAL(HPGV_TRUE);
        }
        
        if (theSwapTimingSaveGlobal) {
            HPGV_TIMING_SAVEGLOBAL(HPGV_TRUE);        
        }
        
        if (theSwapTimingShowGlobal) {
            HPGV_TIMING_SHOWGLOBAL(HPGV_TRUE);
        }
    }
                          
    /* upadte swap control */
    swap_control_update(theSwapControl, width, height, format, type, 
                        partialpixels, finalpixels, depth, root, mpicomm,
                        composite_type);

    /* ========= timing ===========*/
    HPGV_TIMING_COUNTROOT(theSwapControl->roothasimage);
    
    if (composite_type == HPGV_TTSWAP) {
        sprintf(context, "proc%05d_img%dx%d_%s",
                theSwapControl->workernum, width, height, "ttswap");        
    } else if (composite_type == HPGV_DRSEND) {
        sprintf(context, "proc%05d_img%dx%d_%s",
                theSwapControl->workernum, width, height, "drsend");        
    }
    
    HPGV_TIMING_CONTEXTLOCAL(context);
    HPGV_TIMING_CONTEXTGLOBAL(context);         

    /* compositing */
    swap_control_composite(theSwapControl);


    
    return HPGV_SUCCESS;
}

}
