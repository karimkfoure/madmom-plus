/* Shared-library visibility for the native C interfaces. */
#ifndef MADMOM_EXPORTS_H
#define MADMOM_EXPORTS_H

#if defined(_WIN32)
#  if defined(MADMOM_NATIVE_BUILD)
#    define MM_EXPORT __declspec(dllexport)
#  else
#    define MM_EXPORT __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define MM_EXPORT __attribute__((visibility("default")))
#else
#  define MM_EXPORT
#endif

#endif
