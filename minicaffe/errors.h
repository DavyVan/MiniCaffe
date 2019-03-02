/***
 * @file errors.h
 * @author Quan Fan
 * @brief Definitions of error numbers and related functions.
 * @date 01/Mar/2019
 */

/**
 * Cannot use errno as variable name due to a confliction with errno.h
 */

#ifndef _ERRORS_H_
#define _ERRORS_H_

const int ZERO_DIM = 1;         /**< Blobs with any dimension that equals to zero */
const int LEFT_NOT_MATCH = 2;   /**< Expected left blobs don't match any existed blob. */

char* get_err_str(int _errno);

void print_err_str(int _errno);

#endif