/**
* MIT License 
* Copyright (c) 2021-Today Kamil Rog
*
*/

// Header Guard - See One Definition Rule - https://en.cppreference.com/w/cpp/language/definition for details
#ifndef SIMD_LIB
#define SIMD_LIB

// Includes //

#include <cstdio>
#include <cstdint>
#include <type_traits>

// Definitions //

#ifdef __OPTIMIZE__
    #include <immintrin.h>
    // Force compiler to inline
    #define KEWB_FORCE_INLINE inline __atribute__((__always_inline__))
#else
    #define __OPTIMIZE__
    #include <immintrin.h>
    #undef __OPTIMIZE__
    // When building in debug mode use ordinary inline
    #define KEWB_FORCE_INLINE inline
#endif


namespace simdlib
{
    // Register Definitions 
    using rf_512 = __m512;
    using ri_512 = __m512i;
    using msk_512 = uint32_t;


    // Fill Functions //


    /**
    * Fill vector with a 32-bit float 
    */
    KEWB_FORCE_INLINE rf_512
    LoadValue(float fill)
    {
        return _mm512_set1_ps(fill);
    }
    

    /**
    * Fill vector with a 32-bit integer
    */
    KEWB_FORCE_INLINE ri_512
    LoadValue(int32_t fill)
    {
        return _mm512_set1_epi32(fill);
    }


    // Load Functions //


    /**
    * Load from memory for floating point values
    */
    KEWB_FORCE_INLINE rf_512
    LoadFrom(float const* psrc)
    {
        // Use unalligned stores
        // This allows to load from any arbitrarry 32-bit float adress adress
        return _mm512_loadu_ps(psrc);
    }


    /**
    * Load from memory for integer values
    */
    KEWB_FORCE_INLINE ri_512
    LoadFrom(int32_t const* psrc)
    {
        // Use unalligned stores
        // This allows to load from any arbitrarry 32-bit integer adress adress
        return _mm512_loadu_epi32(psrc);
    }


    // Masked Load Functions //

    /**
    * Load from memory for float values using mask and fill value
    */
    KEWB_FORCE_INLINE rf_512
    MaskedLoadFrom(float const* psrc, float fill, msk_512 mask)
    {
        // For set poistion in mask the fill value is loaded, otherwise memory indicated by pointer 
        return _mm512_mask_load_ps(_mm512_set1_ps(fill), (__mmask16) mask, psrc);
    }


    /**
    * Load from memory for integer values using mask and fill value
    */
    KEWB_FORCE_INLINE ri_512
    MaskedLoadFrom(int32_t const* psrc, float fill, msk_512 mask)
    {
        // For set poistion in mask the fill value is loaded, otherwise memory indicated by pointer 
        return _mm512_mask_loadu_epi32(_mm512_set1_epi32(fill), (__mmask16) mask, psrc);
    }


    /**
    * Load from memory for float values using mask and fill value
    */
    KEWB_FORCE_INLINE rf_512
    MaskedLoadFrom(float const* psrc, rf_512 fill, msk_512 mask)
    {
        // For set poistion in mask the fill vector is loaded, otherwise memory indicated by pointer 
        return _mm512_mask_load_ps(fill, (__mmask16) mask, psrc);
    }


    /**
    * Load from memory for integer values using mask and fill value
    */
    KEWB_FORCE_INLINE ri_512
    MaskedLoadFrom(int32_t const* psrc, ri_512 fill, msk_512 mask)
    {
        // For set poistion in mask the fill vector is loaded, otherwise memory indicated by pointer 
        return _mm512_mask_loadu_epi32(fill, (__mmask16) mask, psrc);
    }


    // Store Functions //
    

    /**
    * Store vector of floating points to memory
    */
    KEWB_FORCE_INLINE void
    StoreTo(float *pdst, rf_512 vec)
    {
       _mm512_mask_storeu_ps(pdst, (__mmask16) 0xFFFFu, vec);
    }


    /**
    * Store vector of integers to memory
    */
    KEWB_FORCE_INLINE void
    StoreTo(int32_t *pdst, ri_512 vec)
    {
       _mm512_mask_storeu_epi32(pdst, (__mmask16) 0xFFFFu, vec);
    }


    // Masked Store Functions //


    /**
    * Masked Store vector of floating points to memory
    */
    KEWB_FORCE_INLINE void
    MaskedStoreTo(float *pdst, rf_512 vec, msk_512 mask)
    {
        // For
       _mm512_mask_storeu_ps(pdst, (__mmask16) mask, vec);
    }


    /**
    * Masked Store vector of integers to memory
    */
    KEWB_FORCE_INLINE void
    MaskedStoreTo(int32_t *pdst, ri_512 vec, msk_512 mask)
    {
       _mm512_mask_storeu_epi32(pdst, (__mmask16) mask, vec);
    }


    // Bit Mask //
    

    /**
    * Create a mask
    */
    template<   unsigned A=0, unsigned B=0, unsigned C=0, unsigned D=0,
                unsigned E=0, unsigned F=0, unsigned G=0, unsigned H=0,
                unsigned I=0, unsigned J=0, unsigned K=0, unsigned L=0,
                unsigned M=0, unsigned N=0, unsigned O=0, unsigned P=0 >
    KEWB_FORCE_INLINE constexpr uint32_t
    MakeBitMask()
    {
        // In vector format     = A B C D E F G H I J K L M N O P
        // In big endian mask   = 0bPONM LKJI HGFE DCBA

        static_assert(  (A < 2) && (B < 2) && (C < 2) && (D < 2) &&
                        (E < 2) && (F < 2) && (G < 2) && (H < 2) &&
                        (I < 2) && (J < 2) && (K < 2) && (L < 2) &&
                        (M < 2) && (N < 2) && (O < 2) && (P < 2) );

        return ((A <<  0) | (B <<  1) | (C <<  2) | (D <<  3) |
                (E <<  4) | (F <<  5) | (G <<  6) | (H <<  7) |
                (I <<  8) | (J <<  9) | (K << 10) | (L << 11) |
                (M << 12) | (N << 13) | (O << 14) | (P << 15) );
    }


    /**
    * Creates a register with by setting each element source from one of two registeres
    * indicated by the mask 
    */
    KEWB_FORCE_INLINE rf_512
    Blend(rf_512 a, rf_512 b, msk_512 mask)
    {
        // For each set location in mask the function sets a, otherwise b
        return _mm512_mask_blend_ps( (__mmask16) mask, a, b);
    }


    /**
    * Permutes the register based on the permuatation sequence
    */
    KEWB_FORCE_INLINE rf_512
    Permute(rf_512 r, ri_512 perm)
    {
        // perm locations values are the locations to load the locations from the register
        return _mm512_permutexvar_ps(perm, r);
    }


    /**
    * Permutes the register based on the permuatation sequence
    */
    KEWB_FORCE_INLINE rf_512
    MaskedPermute(rf_512 a, rf_512 b, ri_512 perm, msk_512 mask)
    {
        // For each set location in mask the function sets a, otherwise b
        return _mm512_mask_permutexvar_ps(a, (__mmask16) mask, perm, b);
    }


    /**
    * Make a permutation map
    */
    template<   unsigned A=0, unsigned B=0, unsigned C=0, unsigned D=0,
                unsigned E=0, unsigned F=0, unsigned G=0, unsigned H=0,
                unsigned I=0, unsigned J=0, unsigned K=0, unsigned L=0,
                unsigned M=0, unsigned N=0, unsigned O=0, unsigned P=0 >
    KEWB_FORCE_INLINE ri_512
    MakePermMap()
    {
        static_assert(  (A < 16) && (B < 16) && (C < 16) && (D < 16) &&
                        (E < 16) && (F < 16) && (G < 16) && (H < 16) &&
                        (I < 16) && (J < 16) && (K < 16) && (L < 16) &&
                        (M < 16) && (N < 16) && (O < 16) && (P < 16) );

        // In vector format     = A B C D E F G H I J K L M N O P
        return _mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P);
    }



    /**
    * Rotates the elements in the register based on the rotation factor
    */
    template<int R> KEWB_FORCE_INLINE rf_512
    Rotate(rf_512 r)
    {
        // If rotation factor is equal to the size of the register
        if constexpr((R % 16) == 0)
        {
            // return the register
            return r;
        }
        // Rotate
        else
        {
            constexpr int S = (R > 0) ? (16 - (R % 16)) : R
            constexpr int A = (S +  0) % 16;
            constexpr int B = (S +  1) % 16;
            constexpr int C = (S +  2) % 16;
            constexpr int D = (S +  3) % 16;
            constexpr int E = (S +  4) % 16;
            constexpr int F = (S +  5) % 16;
            constexpr int G = (S +  6) % 16;
            constexpr int H = (S +  7) % 16;
            constexpr int I = (S +  8) % 16;
            constexpr int J = (S +  9) % 16;
            constexpr int K = (S + 10) % 16;
            constexpr int L = (S + 11) % 16;
            constexpr int M = (S + 12) % 16;
            constexpr int N = (S + 13) % 16;
            constexpr int O = (S + 14) % 16;
            constexpr int P = (S + 15) % 16;

            return _mm512_permutexvar_ps(_mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P), r);
        }
    }


    /**
    * Rotates the elements in the register down
    */
    template<int R> KEWB_FORCE_INLINE rf_512
    Rotate(rf_512 r)
    {
        // If rotation factor is equal to the size of the register
        if constexpr((R % 16) == 0)
        {
            // return the register
            return r;
        }
        // Rotate
        else
        {
            constexpr int S = (R > 0) ? (16 - (R % 16)) : R
            constexpr int A = (S +  0) % 16;
            constexpr int B = (S +  1) % 16;
            constexpr int C = (S +  2) % 16;
            constexpr int D = (S +  3) % 16;
            constexpr int E = (S +  4) % 16;
            constexpr int F = (S +  5) % 16;
            constexpr int G = (S +  6) % 16;
            constexpr int H = (S +  7) % 16;
            constexpr int I = (S +  8) % 16;
            constexpr int J = (S +  9) % 16;
            constexpr int K = (S + 10) % 16;
            constexpr int L = (S + 11) % 16;
            constexpr int M = (S + 12) % 16;
            constexpr int N = (S + 13) % 16;
            constexpr int O = (S + 14) % 16;
            constexpr int P = (S + 15) % 16;

            return _mm512_permutexvar_ps(_mm512_setr_epi32(A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P), r);
        }
    }


}

#endif // SIMD_LIB
