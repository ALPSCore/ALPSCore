/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* This File:
* Copyright (C) 2006 by Andreas Laeuchli <laeuchli@comp-phys.ch>,
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/*
 * Sun RPC is a product of Sun Microsystems, Inc. and is provided for
 * unrestricted use provided that this legend is included on all tape
 * media and as a part of the software program in whole or part.  Users
 * may copy or modify Sun RPC without charge, but are not authorized
 * to license or distribute it to anyone else except as part of a product or
 * program developed by the user.
 *
 * SUN RPC IS PROVIDED AS IS WITH NO WARRANTIES OF ANY KIND INCLUDING THE
 * WARRANTIES OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE, OR ARISING FROM A COURSE OF DEALING, USAGE OR TRADE PRACTICE.
 *
 * Sun RPC is provided with no support and without any obligation on the
 * part of Sun Microsystems, Inc. to assist in its use, correction,
 * modification or enhancement.
 *
 * SUN MICROSYSTEMS, INC. SHALL HAVE NO LIABILITY WITH RESPECT TO THE
 * INFRINGEMENT OF COPYRIGHTS, TRADE SECRETS OR ANY PATENTS BY SUN RPC
 * OR ANY PART THEREOF.
 *
 * In no event will Sun Microsystems, Inc. be liable for any lost revenue
 * or profits or other special, indirect and consequential damages, even if
 * Sun has been advised of the possibility of such damages.
 *
 * Sun Microsystems, Inc.
 * 2550 Garcia Avenue
 * Mountain View, California  94043
 *
 *      from: @(#)xdr.h 1.19 87/04/22 SMI
 *      from: @(#)xdr.h 2.2 88/07/29 4.0 RPCSRC
 * $FreeBSD: src/include/rpc/xdr.h,v 1.23 2003/03/07 13:19:40 nectar Exp $
 */

#include <alps/config.h>

#ifndef ALPS_HAVE_RPC_XDR_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <alps/osiris/xdrcore.h>

/*
 * constants specific to the xdr "protocol"
 */
#define XDR_FALSE	((long) 0)
#define XDR_TRUE	((long) 1)
#define LASTUNSIGNED	((u_int) 0-1)

/*
 * for unit alignment
 */
static const char xdr_zero[BYTES_PER_XDR_UNIT] = {0, 0, 0, 0};

/*
 some arpa/inet little/big-endian conversion routines
 */

#if !defined(__BIG_ENDIAN) || !defined(__LITTLE_ENDIAN) || !defined(__BYTE_ORDER)
#error No endianess given
#endif

#if  __BYTE_ORDER == __BIG_ENDIAN
unsigned long
htonl(unsigned long a)
{

        return (a);
}

unsigned long
ntohl(unsigned long a)
{

        return (a);
}

#else 
#if __BYTE_ORDER == __LITTLE_ENDIAN 
unsigned long
htonl(unsigned long a)
{
 return ((((unsigned long)(a) & 0xff000000) >> 24) | \
         (((unsigned long)(a) & 0x00ff0000) >> 8)  | \
         (((unsigned long)(a) & 0x0000ff00) << 8)  | \
         (((unsigned long)(a) & 0x000000ff) << 24));
}
unsigned long
ntohl(unsigned long a)
{
 return ((((unsigned long)(a) & 0xff000000) >> 24) | \
         (((unsigned long)(a) & 0x00ff0000) >> 8)  | \
         (((unsigned long)(a) & 0x0000ff00) << 8)  | \
         (((unsigned long)(a) & 0x000000ff) << 24));
}
#endif
#endif

/*
 * XDR integers
 */
bool_t
xdr_int (XDR *xdrs, int *ip)
{

#if INT_MAX < LONG_MAX
  long l;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      l = (long) *ip;
      return XDR_PUTLONG (xdrs, &l);

    case XDR_DECODE:
      if (!XDR_GETLONG (xdrs, &l))
	{
	  return FALSE;
	}
      *ip = (int) l;
    case XDR_FREE:
      return TRUE;
    }
  return FALSE;
#elif INT_MAX == LONG_MAX
  return xdr_long(xdrs, (long *) ip);
#elif INT_MAX == SHRT_MAX
  return xdr_short(xdrs, (short *) ip);
#else
#error unexpected integer sizes in_xdr_int()
#endif
}

/*
 * XDR unsigned integers
 */
bool_t
xdr_u_int (XDR *xdrs, u_int *up)
{
#if UINT_MAX < ULONG_MAX
  long l;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      l = (u_long) * up;
      return XDR_PUTLONG (xdrs, &l);

    case XDR_DECODE:
      if (!XDR_GETLONG (xdrs, &l))
	{
	  return FALSE;
	}
      *up = (u_int) (u_long) l;
    case XDR_FREE:
      return TRUE;
    }
  return FALSE;
#elif UINT_MAX == ULONG_MAX
  return xdr_u_long(xdrs, (u_long *) up);
#elif UINT_MAX == USHRT_MAX
  return xdr_short(xdrs, (short *) up);
#else
#error unexpected integer sizes in_xdr_u_int()
#endif
}

/*
 * XDR short integers
 */
bool_t
xdr_short (XDR *xdrs, short *sp)
{
  long l;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      l = (long) *sp;
      return XDR_PUTLONG (xdrs, &l);

    case XDR_DECODE:
      if (!XDR_GETLONG (xdrs, &l))
	{
	  return FALSE;
	}
      *sp = (short) l;
      return TRUE;

    case XDR_FREE:
      return TRUE;
    }
  return FALSE;
}

/*
 * XDR unsigned short integers
 */
bool_t
xdr_u_short (XDR *xdrs, u_short *usp)
{
  long l;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      l = (u_long) * usp;
      return XDR_PUTLONG (xdrs, &l);

    case XDR_DECODE:
      if (!XDR_GETLONG (xdrs, &l))
	{
	  return FALSE;
	}
      *usp = (u_short) (u_long) l;
      return TRUE;

    case XDR_FREE:
      return TRUE;
    }
  return FALSE;
}

/*
 * XDR long integers
 * The definition of xdr_long() is kept for backward
 * compatibility. Instead xdr_int() should be used.
 */
bool_t
xdr_long (XDR *xdrs, long *lp)
{

  if (xdrs->x_op == XDR_ENCODE
      && (sizeof (int32_t) == sizeof (long)
	  || (int32_t) *lp == *lp))
    return XDR_PUTLONG (xdrs, lp);

  if (xdrs->x_op == XDR_DECODE)
    return XDR_GETLONG (xdrs, lp);

  if (xdrs->x_op == XDR_FREE)
    return TRUE;

  return FALSE;
}

/*
 * XDR unsigned long integers
 * The definition of xdr_u_long() is kept for backward
 * compatibility. Instead xdr_u_int() should be used.
 */
bool_t
xdr_u_long (XDR *xdrs, u_long *ulp)
{
  switch (xdrs->x_op)
    {
    case XDR_DECODE:
      {
	long int tmp;

	if (XDR_GETLONG (xdrs, &tmp) == FALSE)
	  return FALSE;

	*ulp = (uint32_t) tmp;
	return TRUE;
      }

    case XDR_ENCODE:
      if (sizeof (uint32_t) != sizeof (u_long)
	  && (uint32_t) *ulp != *ulp)
	return FALSE;

      return XDR_PUTLONG (xdrs, (long *) ulp);

    case XDR_FREE:
      return TRUE;
    }
  return FALSE;
}

/*
 * XDR a char
 */
bool_t
xdr_char (XDR *xdrs, char *cp)
{
  int i;

  i = (*cp);
  if (!xdr_int(xdrs, &i))
    {
      return FALSE;
    }
  *cp = i;
  return TRUE;
}

/* 
 * XDR an unsigned char
 */
bool_t
xdr_u_char (XDR *xdrs, u_char *cp)
{
  u_int u; 
 
  u = (*cp);
  if (!xdr_u_int(xdrs, &u))
    {
      return FALSE;
    }
  *cp = u;
  return TRUE;
}


/*
 * XDR booleans
 */
bool_t
xdr_bool (XDR *xdrs, bool_t *bp)
{
  long lb;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      lb = *bp ? XDR_TRUE : XDR_FALSE;
      return XDR_PUTLONG (xdrs, &lb);

    case XDR_DECODE:
      if (!XDR_GETLONG (xdrs, &lb))
	{
	  return FALSE;
	}
      *bp = (lb == XDR_FALSE) ? FALSE : TRUE;
      return TRUE;

    case XDR_FREE:
      return TRUE;
    }
  return FALSE;
}

/*
 * XDR opaque data
 * Allows the specification of a fixed size sequence of opaque bytes.
 * cp points to the opaque object and cnt gives the byte length.
 */
bool_t
xdr_opaque (XDR *xdrs, caddr_t cp, u_int cnt)
{
  u_int rndup;
  static char crud[BYTES_PER_XDR_UNIT];

  /*
   * if no data we are done
   */
  if (cnt == 0)
    return TRUE;

  /*
   * round byte count to full xdr units
   */
  rndup = cnt % BYTES_PER_XDR_UNIT;
  if (rndup > 0)
    rndup = BYTES_PER_XDR_UNIT - rndup;

  switch (xdrs->x_op)
    {
    case XDR_DECODE:
      if (!XDR_GETBYTES (xdrs, cp, cnt))
	{
	  return FALSE;
	}
      if (rndup == 0)
	return TRUE;
      return XDR_GETBYTES (xdrs, (caddr_t)crud, rndup);

    case XDR_ENCODE:
      if (!XDR_PUTBYTES (xdrs, cp, cnt))
	{
	  return FALSE;
	}
      if (rndup == 0)
	return TRUE;
      return XDR_PUTBYTES (xdrs, xdr_zero, rndup);

    case XDR_FREE:
      return TRUE;
    }
  return FALSE;
}

/*
 * XDR null terminated ASCII strings
 * xdr_string deals with "C strings" - arrays of bytes that are
 * terminated by a NULL character.  The parameter cpp references a
 * pointer to storage; If the pointer is null, then the necessary
 * storage is allocated.  The last parameter is the max allowed length
 * of the string as specified by a protocol.
 */

bool_t xdr_string (XDR *xdrs,char ** cpp,u_int maxsize)
{
  char *sp = *cpp;	/* sp is the actual string pointer */
  u_int size;
  u_int nodesize;

  /*
   * first deal with the length since xdr strings are counted-strings
   */
  switch (xdrs->x_op)
    {
    case XDR_FREE:
      if (sp == NULL)
	{
	  return TRUE;		/* already free */
	}
      /* fall through... */
    case XDR_ENCODE:
      if (sp == NULL)
	return FALSE;
      size = strlen (sp);
      break;
    case XDR_DECODE:
      break;
    }
  if (!xdr_u_int(xdrs, &size))
    {
      return FALSE;
    }
  if (size > maxsize)
    {
      return FALSE;
    }
  nodesize = size + 1;
  if (nodesize == 0)
    {
      /* This means an overflow.  It a bug in the caller which
	 provided a too large maxsize but nevertheless catch it
	 here.  */
      return FALSE;
    }

  /*
   * now deal with the actual bytes
   */
  switch (xdrs->x_op)
    {
    case XDR_DECODE:
      if (sp == NULL)
	*cpp = sp = (char *) malloc (nodesize);
      if (sp == NULL)
	{
	  (void) printf ("%s", "xdr_string: out of memory\n");
	  return FALSE;
	}
      sp[size] = 0;
      /* fall into ... */

    case XDR_ENCODE:
      return xdr_opaque(xdrs, sp, size);

    case XDR_FREE:
      free (sp);
      *cpp = NULL;
      return TRUE;
    }
  return FALSE;
}

#ifndef __FLOAT_WORD_ORDER
#define __FLOAT_WORD_ORDER __BYTE_ORDER
#endif

#define LSW     (__FLOAT_WORD_ORDER == __BIG_ENDIAN)

#ifdef vax

/* What IEEE single precision floating point looks like on a Vax */
struct  ieee_single {
        unsigned int    mantissa: 23;
        unsigned int    exp     : 8;
        unsigned int    sign    : 1;
};

/* Vax single precision floating point */
struct  vax_single {
        unsigned int    mantissa1 : 7;
        unsigned int    exp       : 8;
        unsigned int    sign      : 1;
        unsigned int    mantissa2 : 16;
};

#define VAX_SNG_BIAS    0x81
#define IEEE_SNG_BIAS   0x7f

static struct sgl_limits {
        struct vax_single s;
        struct ieee_single ieee;
} sgl_limits[2] = {
        {{ 0x7f, 0xff, 0x0, 0xffff },   /* Max Vax */
        { 0x0, 0xff, 0x0 }},            /* Max IEEE */
        {{ 0x0, 0x0, 0x0, 0x0 },        /* Min Vax */
        { 0x0, 0x0, 0x0 }}              /* Min IEEE */
};
#endif /* vax */

bool_t
xdr_float(XDR* xdrs,float* fp)
{
#ifdef vax
        struct ieee_single is;
        struct vax_single vs, *vsp;
        struct sgl_limits *lim;
        int i;
#endif
        switch (xdrs->x_op) {

        case XDR_ENCODE:
#ifdef vax
                vs = *((struct vax_single *)fp);
                for (i = 0, lim = sgl_limits;
                        i < sizeof(sgl_limits)/sizeof(struct sgl_limits);
                        i++, lim++) {
                        if ((vs.mantissa2 == lim->s.mantissa2) &&
                                (vs.exp == lim->s.exp) &&
                                (vs.mantissa1 == lim->s.mantissa1)) {
                                is = lim->ieee;
                                goto shipit;
                        }
                }
                is.exp = vs.exp - VAX_SNG_BIAS + IEEE_SNG_BIAS;
                is.mantissa = (vs.mantissa1 << 16) | vs.mantissa2;
        shipit:
                is.sign = vs.sign;
                return (XDR_PUTLONG(xdrs, (long *)&is));
#else
                if (sizeof(float) == sizeof(long))
                        return (XDR_PUTLONG(xdrs, (long *)fp));
                else if (sizeof(float) == sizeof(int)) {
                        long tmp = *(int *)fp;
                        return (XDR_PUTLONG(xdrs, &tmp));
                }
                break;
#endif
        case XDR_DECODE:
#ifdef vax
                vsp = (struct vax_single *)fp;
                if (!XDR_GETLONG(xdrs, (long *)&is))
                        return (FALSE);
                for (i = 0, lim = sgl_limits;
                        i < sizeof(sgl_limits)/sizeof(struct sgl_limits);
                        i++, lim++) {
                        if ((is.exp == lim->ieee.exp) &&
                                (is.mantissa == lim->ieee.mantissa)) {
                                *vsp = lim->s;
                                goto doneit;
                        }
                }
                vsp->exp = is.exp - IEEE_SNG_BIAS + VAX_SNG_BIAS;
                vsp->mantissa2 = is.mantissa;
                vsp->mantissa1 = (is.mantissa >> 16);
        doneit:
                vsp->sign = is.sign;
                return (TRUE);
#else
                if (sizeof(float) == sizeof(long))
                        return (XDR_GETLONG(xdrs, (long *)fp));
                else if (sizeof(float) == sizeof(int)) {
                        long tmp;
                        if (XDR_GETLONG(xdrs, &tmp)) {
                                *(int *)fp = tmp;
                                return (TRUE);
                        }
                }
                break;
#endif

        case XDR_FREE:
                return (TRUE);
        }
        return (FALSE);
}



/*
 * This routine works on Suns (Sky / 68000's) and Vaxen.
 */

#ifdef vax
/* What IEEE double precision floating point looks like on a Vax */
struct	ieee_double {
	unsigned int	mantissa1 : 20;
	unsigned int	exp       : 11;
	unsigned int	sign      : 1;
	unsigned int	mantissa2 : 32;
};

/* Vax double precision floating point */
struct  vax_double {
	unsigned int	mantissa1 : 7;
	unsigned int	exp       : 8;
	unsigned int	sign      : 1;
	unsigned int	mantissa2 : 16;
	unsigned int	mantissa3 : 16;
	unsigned int	mantissa4 : 16;
};

#define VAX_DBL_BIAS	0x81
#define IEEE_DBL_BIAS	0x3ff
#define MASK(nbits)	((1 << nbits) - 1)

static struct dbl_limits {
	struct	vax_double d;
	struct	ieee_double ieee;
} dbl_limits[2] = {
	{{ 0x7f, 0xff, 0x0, 0xffff, 0xffff, 0xffff },	/* Max Vax */
	{ 0x0, 0x7ff, 0x0, 0x0 }},			/* Max IEEE */
	{{ 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},		/* Min Vax */
	{ 0x0, 0x0, 0x0, 0x0 }}				/* Min IEEE */
};

#endif /* vax */

bool_t
xdr_double(XDR *xdrs,double * dp)
{
#ifdef vax
	struct	ieee_double id;
	struct	vax_double vd;
	register struct dbl_limits *lim;
	int i;
#endif

	switch (xdrs->x_op) {

	case XDR_ENCODE:
#ifdef vax
		vd = *((struct vax_double *)dp);
		for (i = 0, lim = dbl_limits;
			i < sizeof(dbl_limits)/sizeof(struct dbl_limits);
			i++, lim++) {
			if ((vd.mantissa4 == lim->d.mantissa4) &&
				(vd.mantissa3 == lim->d.mantissa3) &&
				(vd.mantissa2 == lim->d.mantissa2) &&
				(vd.mantissa1 == lim->d.mantissa1) &&
				(vd.exp == lim->d.exp)) {
				id = lim->ieee;
				goto shipit;
			}
		}
		id.exp = vd.exp - VAX_DBL_BIAS + IEEE_DBL_BIAS;
		id.mantissa1 = (vd.mantissa1 << 13) | (vd.mantissa2 >> 3);
		id.mantissa2 = ((vd.mantissa2 & MASK(3)) << 29) |
				(vd.mantissa3 << 13) |
				((vd.mantissa4 >> 3) & MASK(13));
	shipit:
		id.sign = vd.sign;
		dp = (double *)&id;
#endif
		if (2*sizeof(long) == sizeof(double)) {
			long *lp = (long *)dp;
			return (XDR_PUTLONG(xdrs, lp+!LSW) &&
				XDR_PUTLONG(xdrs, lp+LSW));
		} else if (2*sizeof(int) == sizeof(double)) {
			int *ip = (int *)dp;
			long tmp[2];
			tmp[0] = ip[!LSW];
			tmp[1] = ip[LSW];
			return (XDR_PUTLONG(xdrs, tmp) &&
				XDR_PUTLONG(xdrs, tmp+1));
		}
		break;

	case XDR_DECODE:
#ifdef vax
		lp = (long *)&id;
		if (!XDR_GETLONG(xdrs, lp++) || !XDR_GETLONG(xdrs, lp))
			return (FALSE);
		for (i = 0, lim = dbl_limits;
			i < sizeof(dbl_limits)/sizeof(struct dbl_limits);
			i++, lim++) {
			if ((id.mantissa2 == lim->ieee.mantissa2) &&
				(id.mantissa1 == lim->ieee.mantissa1) &&
				(id.exp == lim->ieee.exp)) {
				vd = lim->d;
				goto doneit;
			}
		}
		vd.exp = id.exp - IEEE_DBL_BIAS + VAX_DBL_BIAS;
		vd.mantissa1 = (id.mantissa1 >> 13);
		vd.mantissa2 = ((id.mantissa1 & MASK(13)) << 3) |
				(id.mantissa2 >> 29);
		vd.mantissa3 = (id.mantissa2 >> 13);
		vd.mantissa4 = (id.mantissa2 << 3);
	doneit:
		vd.sign = id.sign;
		*dp = *((double *)&vd);
		return (TRUE);
#else
		if (2*sizeof(long) == sizeof(double)) {
			long *lp = (long *)dp;
			return (XDR_GETLONG(xdrs, lp+!LSW) &&
				XDR_GETLONG(xdrs, lp+LSW));
		} else if (2*sizeof(int) == sizeof(double)) {
			int *ip = (int *)dp;
			long tmp[2];
			if (XDR_GETLONG(xdrs, tmp+!LSW) &&
			    XDR_GETLONG(xdrs, tmp+LSW)) {
				ip[0] = tmp[0];
				ip[1] = tmp[1];
				return (TRUE);
			}
		}
		break;
#endif

	case XDR_FREE:
		return (TRUE);
	}
	return (FALSE);
}


/*
 * xdr_stdio.c, XDR implementation on standard i/o file.
 *
 * Copyright (C) 1984, Sun Microsystems, Inc.
 *
 * This set of routines implements a XDR on a stdio stream.
 * XDR_ENCODE serializes onto the stream, XDR_DECODE de-serializes
 * from the stream.
 */

/* #include <rpc/types.h> */
#include <stdio.h>
/* #include <rpc/xdr.h> */

#ifdef USE_IN_LIBIO
# include <libio/iolibio.h>
# define fflush(s) INTUSE(_IO_fflush) (s)
# define fread(p, m, n, s) INTUSE(_IO_fread) (p, m, n, s)
# define ftell(s) INTUSE(_IO_ftell) (s)
# define fwrite(p, m, n, s) INTUSE(_IO_fwrite) (p, m, n, s)
#endif

static bool_t xdrstdio_getlong (XDR *, long *);
static bool_t xdrstdio_putlong (XDR *, const long *);
static bool_t xdrstdio_getbytes (XDR *, caddr_t, u_int);
static bool_t xdrstdio_putbytes (XDR *, const char *, u_int);
static u_int xdrstdio_getpos (const XDR *);
static bool_t xdrstdio_setpos (XDR *, u_int);
static int32_t *xdrstdio_inline (XDR *, u_int);
static void xdrstdio_destroy (XDR *);
static bool_t xdrstdio_getint32 (XDR *, int32_t *);
static bool_t xdrstdio_putint32 (XDR *, const int32_t *);

/*
 * Ops vector for stdio type XDR
 */
static const struct XDR::xdr_ops xdrstdio_ops =
{
  xdrstdio_getlong,		/* deserialize a long int */
  xdrstdio_putlong,		/* serialize a long int */
  xdrstdio_getbytes,		/* deserialize counted bytes */
  xdrstdio_putbytes,		/* serialize counted bytes */
  xdrstdio_getpos,		/* get offset in the stream */
  xdrstdio_setpos,		/* set offset in the stream */
  xdrstdio_inline,		/* prime stream for inline macros */
  xdrstdio_destroy,		/* destroy stream */
  xdrstdio_getint32,		/* deserialize a int */
  xdrstdio_putint32		/* serialize a int */
};

/*
 * Initialize a stdio xdr stream.
 * Sets the xdr stream handle xdrs for use on the stream file.
 * Operation flag is set to op.
 */
void
xdrstdio_create (XDR *xdrs, FILE *file, enum xdr_op op)
{
  xdrs->x_op = op;
  /* We have to add the const since the `struct xdr_ops' in `struct XDR'
     is not `const'.  */
  xdrs->x_ops = (struct XDR::xdr_ops *) &xdrstdio_ops;
  xdrs->x_private = (caddr_t) file;
  xdrs->x_handy = 0;
  xdrs->x_base = 0;
}

/*
 * Destroy a stdio xdr stream.
 * Cleans up the xdr stream handle xdrs previously set up by xdrstdio_create.
 */
static void
xdrstdio_destroy (XDR *xdrs)
{
  (void) fflush ((FILE *) xdrs->x_private);
  /* xx should we close the file ?? */
};

static bool_t
xdrstdio_getlong (XDR *xdrs, long *lp)
{
  u_int32_t mycopy;

  if (fread ((caddr_t) &mycopy, 4, 1, (FILE *) xdrs->x_private) != 1)
    return FALSE;
  *lp = (long) ntohl (mycopy);
  return TRUE;
}

static bool_t
xdrstdio_putlong (XDR *xdrs, const long *lp)
{
  int32_t mycopy = htonl ((u_int32_t) *lp);

  if (fwrite ((caddr_t) &mycopy, 4, 1, (FILE *) xdrs->x_private) != 1)
    return FALSE;
  return TRUE;
}

static bool_t
xdrstdio_getbytes (XDR *xdrs, const caddr_t addr, u_int len)
{
  if ((len != 0) && (fread (addr, (int) len, 1,
			    (FILE *) xdrs->x_private) != 1))
    return FALSE;
  return TRUE;
}

static bool_t
xdrstdio_putbytes (XDR *xdrs, const char *addr, u_int len)
{
  if ((len != 0) && (fwrite (addr, (int) len, 1,
			     (FILE *) xdrs->x_private) != 1))
    return FALSE;
  return TRUE;
}

static u_int
xdrstdio_getpos (const XDR *xdrs)
{
  return (u_int) ftell ((FILE *) xdrs->x_private);
}

static bool_t
xdrstdio_setpos (XDR *xdrs, u_int pos)
{
  return fseek ((FILE *) xdrs->x_private, (long) pos, 0) < 0 ? FALSE : TRUE;
}

static int32_t *
xdrstdio_inline (XDR *xdrs, u_int len)
{
  /*
   * Must do some work to implement this: must insure
   * enough data in the underlying stdio buffer,
   * that the buffer is aligned so that we can indirect through a
   * long *, and stuff this pointer in xdrs->x_buf.  Doing
   * a fread or fwrite to a scratch buffer would defeat
   * most of the gains to be had here and require storage
   * management on this buffer, so we don't do this.
   */
  return NULL;
}

static bool_t
xdrstdio_getint32 (XDR *xdrs, int32_t *ip)
{
  int32_t mycopy;

  if (fread ((caddr_t) &mycopy, 4, 1, (FILE *) xdrs->x_private) != 1)
    return FALSE;
  *ip = ntohl (mycopy);
  return TRUE;
}

static bool_t
xdrstdio_putint32 (XDR *xdrs, const int32_t *ip)
{
  int32_t mycopy = htonl (*ip);

  ip = &mycopy;
  if (fwrite ((caddr_t) ip, 4, 1, (FILE *) xdrs->x_private) != 1)
    return FALSE;
  return TRUE;
}

/*
 * xdr_vector():
 *
 * XDR a fixed length array. Unlike variable-length arrays,
 * the storage of fixed length arrays is static and unfreeable.
 * > basep: base of the array
 * > size: size of the array
 * > elemsize: size of each element
 * > xdr_elem: routine to XDR each element
 */

bool_t
xdr_vector (XDR* xdrs,char* basep,u_int nelem,u_int elemsize,xdrproc_t xdr_elem)
{
  u_int i;
  char *elptr;

  elptr = basep;
  for (i = 0; i < nelem; i++)
    {
      if (!(*xdr_elem) (xdrs, elptr, LASTUNSIGNED))
        {
          return FALSE;
        }
      elptr += elemsize;
    }
  return TRUE;
}

#endif /* ALPS_HAVE_RPC_XDR_H */
