/**************************************************************************
 *
 * Copyright (C) 2014 Red Hat Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/
/* create our own EGL offscreen rendering context via gbm and rendernodes */


/* if we are using EGL and rendernodes then we talk via file descriptors to the remote
   node */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define EGL_EGLEXT_PROTOTYPES
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <epoxy/egl.h>
#ifdef HAVE_LIBDRM
#include <xf86drm.h>
#endif

#include "util/u_memory.h"

#include "virglrenderer.h"
#include "virgl_egl.h"
#ifndef EGL_WITHOUT_GBM
#include "virgl_gbm.h"
#endif
#include "virgl_hw.h"
#include "vrend_util.h"

#define EGL_KHR_SURFACELESS_CONTEXT            BIT(0)
#define EGL_KHR_CREATE_CONTEXT                 BIT(1)
#define EGL_MESA_DRM_IMAGE                     BIT(2)
#define EGL_MESA_IMAGE_DMA_BUF_EXPORT          BIT(3)
#define EGL_MESA_DMA_BUF_IMAGE_IMPORT          BIT(4)
#define EGL_KHR_GL_COLORSPACE                  BIT(5)
#define EGL_EXT_IMAGE_DMA_BUF_IMPORT           BIT(6)
#define EGL_EXT_IMAGE_DMA_BUF_IMPORT_MODIFIERS BIT(7)
#define EGL_KHR_FENCE_SYNC                     BIT(8)

static const struct {
   uint32_t bit;
   const char *string;
} extensions_list[] = {
   { EGL_KHR_SURFACELESS_CONTEXT, "EGL_KHR_surfaceless_context" },
   { EGL_KHR_CREATE_CONTEXT, "EGL_KHR_create_context" },
   { EGL_MESA_DRM_IMAGE, "EGL_MESA_drm_image" },
   { EGL_MESA_IMAGE_DMA_BUF_EXPORT, "EGL_MESA_image_dma_buf_export" },
   { EGL_KHR_GL_COLORSPACE, "EGL_KHR_gl_colorspace" },
   { EGL_EXT_IMAGE_DMA_BUF_IMPORT, "EGL_EXT_image_dma_buf_import" },
   { EGL_EXT_IMAGE_DMA_BUF_IMPORT_MODIFIERS, "EGL_EXT_image_dma_buf_import_modifiers" },
   { EGL_KHR_FENCE_SYNC, "EGL_KHR_fence_sync"}
};

struct virgl_egl {
   struct virgl_gbm *gbm;
   EGLDisplay egl_display;
   EGLConfig egl_conf;
   EGLContext egl_ctx;
   uint32_t extension_bits;
   bool need_fence_and_wait_external;
};

static bool virgl_egl_has_extension_in_string(const char *haystack, const char *needle)
{
   const unsigned needle_len = strlen(needle);

   if (needle_len == 0)
      return false;

   while (true) {
      const char *const s = strstr(haystack, needle);

      if (s == NULL)
         return false;

      if (s[needle_len] == ' ' || s[needle_len] == '\0') {
         return true;
      }

      /* strstr found an extension whose name begins with
       * needle, but whose name is not equal to needle.
       * Restart the search at s + needle_len so that we
       * don't just find the same extension again and go
       * into an infinite loop.
       */
      haystack = s + needle_len;
   }

   return false;
}

static int virgl_egl_init_extensions(struct virgl_egl *egl, const char *extensions)
{
   for (uint32_t i = 0; i < ARRAY_SIZE(extensions_list); i++) {
      if (virgl_egl_has_extension_in_string(extensions, extensions_list[i].string))
         egl->extension_bits |= extensions_list[i].bit;
   }

   if (!has_bits(egl->extension_bits, EGL_KHR_SURFACELESS_CONTEXT | EGL_KHR_CREATE_CONTEXT)) {
      vrend_printf( "Missing EGL_KHR_surfaceless_context or EGL_KHR_create_context\n");
      return -1;
   }

   return 0;
}

struct virgl_egl *virgl_egl_init(struct virgl_gbm *gbm, bool surfaceless, bool gles)
{
   static EGLint conf_att[] = {
      EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
      EGL_RED_SIZE, 1,
      EGL_GREEN_SIZE, 1,
      EGL_BLUE_SIZE, 1,
      EGL_ALPHA_SIZE, 0,
      EGL_NONE,
   };
   static const EGLint ctx_att[] = {
      EGL_CONTEXT_CLIENT_VERSION, 2,
      EGL_NONE
   };
   EGLBoolean success;
   EGLenum api;
   EGLint major, minor, num_configs;
   const char *extensions;
   struct virgl_egl *egl;

   egl = calloc(1, sizeof(struct virgl_egl));
   if (!egl)
      return NULL;

   if (gles)
      conf_att[3] = EGL_OPENGL_ES2_BIT;

   if (surfaceless)
      conf_att[1] = EGL_PBUFFER_BIT;
#ifndef EGL_WITHOUT_GBM
   else if (!gbm)
      goto fail;

   egl->gbm = gbm;
#endif
   const char *client_extensions = eglQueryString (NULL, EGL_EXTENSIONS);
#ifndef EGL_WITHOUT_GBM
   if (client_extensions && strstr(client_extensions, "EGL_KHR_platform_base")) {
      PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
         (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress ("eglGetPlatformDisplay");

      if (!get_platform_display)
        goto fail;

      if (surfaceless) {
         egl->egl_display = get_platform_display (EGL_PLATFORM_SURFACELESS_MESA,
                                                  EGL_DEFAULT_DISPLAY, NULL);
      } else
         egl->egl_display = get_platform_display (EGL_PLATFORM_GBM_KHR,
                                                  (EGLNativeDisplayType)egl->gbm->device, NULL);
   } else if (client_extensions && strstr(client_extensions, "EGL_EXT_platform_base")) {
      PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
         (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress ("eglGetPlatformDisplayEXT");

      if (!get_platform_display)
        goto fail;

      if (surfaceless) {
         egl->egl_display = get_platform_display (EGL_PLATFORM_SURFACELESS_MESA,
                                                  EGL_DEFAULT_DISPLAY, NULL);
      } else
         egl->egl_display = get_platform_display (EGL_PLATFORM_GBM_KHR,
                                                 (EGLNativeDisplayType)egl->gbm->device, NULL);
   } else {
      egl->egl_display = eglGetDisplay((EGLNativeDisplayType)egl->gbm->device);
   }
#endif
   if (!egl->egl_display) {
#ifndef EGL_WITHOUT_GBM
      /*
       * Don't fallback to the default display if the fd provided by (*get_drm_fd)
       * can't be used.
       */
      if (egl->gbm && egl->gbm->fd < 0)
         goto fail;
#endif
      egl->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (!egl->egl_display)
         goto fail;
   }

   success = eglInitialize(egl->egl_display, &major, &minor);
   if (!success)
      goto fail;

   extensions = eglQueryString(egl->egl_display, EGL_EXTENSIONS);
#ifdef VIRGL_EGL_DEBUG
   vrend_printf( "EGL major/minor: %d.%d\n", major, minor);
   vrend_printf( "EGL version: %s\n",
           eglQueryString(egl->egl_display, EGL_VERSION));
   vrend_printf( "EGL vendor: %s\n",
           eglQueryString(egl->egl_display, EGL_VENDOR));
   vrend_printf( "EGL extensions: %s\n", extensions);
#endif

   if (virgl_egl_init_extensions(egl, extensions))
      goto fail;

   // ARM Mali platforms need explicit synchronization prior to mapping.
   if (!strcmp(eglQueryString(egl->egl_display, EGL_VENDOR), "ARM"))
      egl->need_fence_and_wait_external = true;

   if (gles)
      api = EGL_OPENGL_ES_API;
   else
      api = EGL_OPENGL_API;
   success = eglBindAPI(api);
   if (!success)
      goto fail;

   success = eglChooseConfig(egl->egl_display, conf_att, &egl->egl_conf,
                             1, &num_configs);
   if (!success || num_configs != 1)
      goto fail;

   egl->egl_ctx = eglCreateContext(egl->egl_display, egl->egl_conf, EGL_NO_CONTEXT,
                                   ctx_att);
   if (!egl->egl_ctx)
      goto fail;

   eglMakeCurrent(egl->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                  egl->egl_ctx);
   return egl;

 fail:
   free(egl);
   return NULL;
}

void virgl_egl_destroy(struct virgl_egl *egl)
{
   eglMakeCurrent(egl->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                  EGL_NO_CONTEXT);
   eglDestroyContext(egl->egl_display, egl->egl_ctx);
   eglTerminate(egl->egl_display);
   free(egl);
}

virgl_renderer_gl_context virgl_egl_create_context(struct virgl_egl *egl, struct virgl_gl_ctx_param *vparams)
{
   EGLContext egl_ctx;
   EGLint ctx_att[] = {
      EGL_CONTEXT_CLIENT_VERSION, vparams->major_ver,
      EGL_CONTEXT_MINOR_VERSION_KHR, vparams->minor_ver,
      EGL_NONE
   };
   egl_ctx = eglCreateContext(egl->egl_display,
                             egl->egl_conf,
                             vparams->shared ? eglGetCurrentContext() : EGL_NO_CONTEXT,
                             ctx_att);
   return (virgl_renderer_gl_context)egl_ctx;
}

void virgl_egl_destroy_context(struct virgl_egl *egl, virgl_renderer_gl_context virglctx)
{
   EGLContext egl_ctx = (EGLContext)virglctx;
   eglDestroyContext(egl->egl_display, egl_ctx);
}

int virgl_egl_make_context_current(struct virgl_egl *egl, virgl_renderer_gl_context virglctx)
{
   EGLContext egl_ctx = (EGLContext)virglctx;

   return eglMakeCurrent(egl->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                         egl_ctx);
}

virgl_renderer_gl_context virgl_egl_get_current_context(UNUSED struct virgl_egl *egl)
{
   EGLContext egl_ctx = eglGetCurrentContext();
   return (virgl_renderer_gl_context)egl_ctx;
}

#ifndef EGL_WITHOUT_GBM
int virgl_egl_get_fourcc_for_texture(struct virgl_egl *egl, uint32_t tex_id, uint32_t format, int *fourcc)
{
   int ret = EINVAL;
   uint32_t gbm_format = 0;

   EGLImageKHR image;
   EGLBoolean success;

   if (!has_bit(egl->extension_bits, EGL_MESA_IMAGE_DMA_BUF_EXPORT)) {
      ret = 0;
      goto fallback;
   }

   image = eglCreateImageKHR(egl->egl_display, eglGetCurrentContext(), EGL_GL_TEXTURE_2D_KHR,
                            (EGLClientBuffer)(unsigned long)tex_id, NULL);

   if (!image)
      return EINVAL;

   success = eglExportDMABUFImageQueryMESA(egl->egl_display, image, fourcc, NULL, NULL);
   if (!success)
      goto out_destroy;
   ret = 0;
 out_destroy:
   eglDestroyImageKHR(egl->egl_display, image);
   return ret;

 fallback:
   ret = virgl_gbm_convert_format(&format, &gbm_format);
   *fourcc = (int)gbm_format;
   return ret;
}

int virgl_egl_get_fd_for_texture2(struct virgl_egl *egl, uint32_t tex_id, int *fd,
                                  int *stride, int *offset)
{
   int ret = EINVAL;
   EGLImageKHR image = eglCreateImageKHR(egl->egl_display, eglGetCurrentContext(),
                                         EGL_GL_TEXTURE_2D_KHR,
                                         (EGLClientBuffer)(unsigned long)tex_id, NULL);
   if (!image)
      return EINVAL;
   if (!has_bit(egl->extension_bits, EGL_MESA_IMAGE_DMA_BUF_EXPORT))
      goto out_destroy;

   if (!eglExportDMABUFImageMESA(egl->egl_display, image, fd,
                                 stride, offset))
      goto out_destroy;

   ret = 0;

out_destroy:
   eglDestroyImageKHR(egl->egl_display, image);
   return ret;
}

int virgl_egl_get_fd_for_texture(struct virgl_egl *egl, uint32_t tex_id, int *fd)
{
   EGLImageKHR image;
   EGLint stride;
   EGLint offset;
   EGLBoolean success;
   int ret;
   image = eglCreateImageKHR(egl->egl_display, eglGetCurrentContext(), EGL_GL_TEXTURE_2D_KHR,
                            (EGLClientBuffer)(unsigned long)tex_id, NULL);

   if (!image)
      return EINVAL;

   ret = EINVAL;
   if (has_bit(egl->extension_bits, EGL_MESA_IMAGE_DMA_BUF_EXPORT)) {
      success = eglExportDMABUFImageMESA(egl->egl_display, image, fd, &stride,
                                         &offset);
      if (!success)
         goto out_destroy;
   } else if (has_bit(egl->extension_bits, EGL_MESA_DRM_IMAGE)) {
      EGLint handle;
      success = eglExportDRMImageMESA(egl->egl_display, image, NULL, &handle,
                                      &stride);

      if (!success)
         goto out_destroy;

      if (!egl->gbm)
         goto out_destroy;

      ret = virgl_gbm_export_fd(egl->gbm->device, handle, fd);
      if (ret < 0)
         goto out_destroy;
   } else {
      goto out_destroy;
   }

   ret = 0;
 out_destroy:
   eglDestroyImageKHR(egl->egl_display, image);
   return ret;
}
#endif

bool virgl_has_egl_khr_gl_colorspace(struct virgl_egl *egl)
{
   return has_bit(egl->extension_bits, EGL_KHR_GL_COLORSPACE);
}

#ifdef ENABLE_GBM_ALLOCATION
void *virgl_egl_image_from_dmabuf(struct virgl_egl *egl, struct gbm_bo *bo)
{
   int ret;
   EGLImageKHR image;
   int fds[VIRGL_GBM_MAX_PLANES] = {-1, -1, -1, -1};
   int num_planes = gbm_bo_get_plane_count(bo);
   // When the bo has 3 planes with modifier support, it requires 37 components.
   EGLint khr_image_attrs[37] = {
      EGL_WIDTH,
      gbm_bo_get_width(bo),
      EGL_HEIGHT,
      gbm_bo_get_height(bo),
      EGL_LINUX_DRM_FOURCC_EXT,
      (int)gbm_bo_get_format(bo),
      EGL_NONE,
   };

   if (num_planes < 0 || num_planes > VIRGL_GBM_MAX_PLANES)
      return (void *)EGL_NO_IMAGE_KHR;

   for (int plane = 0; plane < num_planes; plane++) {
      uint32_t handle = gbm_bo_get_handle_for_plane(bo, plane).u32;
      ret = virgl_gbm_export_fd(egl->gbm->device, handle, &fds[plane]);
      if (ret < 0) {
         vrend_printf( "failed to export plane handle\n");
         image = (void *)EGL_NO_IMAGE_KHR;
         goto out_close;
      }
   }

   size_t attrs_index = 6;
   for (int plane = 0; plane < num_planes; plane++) {
      khr_image_attrs[attrs_index++] = EGL_DMA_BUF_PLANE0_FD_EXT + plane * 3;
      khr_image_attrs[attrs_index++] = fds[plane];
      khr_image_attrs[attrs_index++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT + plane * 3;
      khr_image_attrs[attrs_index++] = gbm_bo_get_offset(bo, plane);
      khr_image_attrs[attrs_index++] = EGL_DMA_BUF_PLANE0_PITCH_EXT + plane * 3;
      khr_image_attrs[attrs_index++] = gbm_bo_get_stride_for_plane(bo, plane);
      if (has_bit(egl->extension_bits, EGL_EXT_IMAGE_DMA_BUF_IMPORT_MODIFIERS)) {
         const uint64_t modifier = gbm_bo_get_modifier(bo);
         khr_image_attrs[attrs_index++] =
         EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT + plane * 2;
         khr_image_attrs[attrs_index++] = modifier & 0xfffffffful;
         khr_image_attrs[attrs_index++] =
         EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT + plane * 2;
         khr_image_attrs[attrs_index++] = modifier >> 32;
      }
   }

   khr_image_attrs[attrs_index++] = EGL_NONE;
   image = eglCreateImageKHR(egl->egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL,
                             khr_image_attrs);

out_close:
   for (int plane = 0; plane < num_planes; plane++)
      close(fds[plane]);

   return (void*)image;
}

void *virgl_egl_aux_plane_image_from_dmabuf(struct virgl_egl *egl, struct gbm_bo *bo, int plane)
{
   int ret;
   EGLImageKHR image = EGL_NO_IMAGE_KHR;
   int fd = -1;

   int bytes_per_pixel = virgl_gbm_get_plane_bytes_per_pixel(bo, plane);
   if (bytes_per_pixel != 1 && bytes_per_pixel != 2)
      return (void *)EGL_NO_IMAGE_KHR;

   uint32_t handle = gbm_bo_get_handle_for_plane(bo, plane).u32;
   ret = drmPrimeHandleToFD(gbm_device_get_fd(egl->gbm->device), handle, DRM_CLOEXEC, &fd);
   if (ret < 0) {
      vrend_printf("failed to export plane handle %d\n", errno);
      return (void *)EGL_NO_IMAGE_KHR;
   }

   EGLint khr_image_attrs[17] = {
      EGL_WIDTH,
      virgl_gbm_get_plane_width(bo, plane),
      EGL_HEIGHT,
      virgl_gbm_get_plane_height(bo, plane),
      EGL_LINUX_DRM_FOURCC_EXT,
      (int) (bytes_per_pixel == 1 ? GBM_FORMAT_R8 : GBM_FORMAT_GR88),
      EGL_DMA_BUF_PLANE0_FD_EXT,
      fd,
      EGL_DMA_BUF_PLANE0_OFFSET_EXT,
      gbm_bo_get_offset(bo, plane),
      EGL_DMA_BUF_PLANE0_PITCH_EXT,
      gbm_bo_get_stride_for_plane(bo, plane),
   };

   if (has_bit(egl->extension_bits, EGL_EXT_IMAGE_DMA_BUF_IMPORT_MODIFIERS)) {
      const uint64_t modifier = gbm_bo_get_modifier(bo);
      khr_image_attrs[12] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT;
      khr_image_attrs[13] = modifier & 0xfffffffful;
      khr_image_attrs[14] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT;
      khr_image_attrs[15] = modifier >> 32;
      khr_image_attrs[16] = EGL_NONE;
   } else {
      khr_image_attrs[12] = EGL_NONE;
   }

   image = eglCreateImageKHR(egl->egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, khr_image_attrs);

   close(fd);
   return (void*)image;
}

void virgl_egl_image_destroy(struct virgl_egl *egl, void *image)
{
   eglDestroyImageKHR(egl->egl_display, image);
}
#endif

bool virgl_egl_need_fence_and_wait_external(struct virgl_egl *egl)
{
   return (egl && egl->need_fence_and_wait_external);
}

void *virgl_egl_fence(struct virgl_egl *egl)
{
   const EGLint attrib_list[] = {EGL_SYNC_CONDITION_KHR,
                                 EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR,
                                 EGL_NONE};
   EGLSyncKHR fence = EGL_NO_SYNC_KHR;

   if (!egl || !has_bit(egl->extension_bits, EGL_KHR_FENCE_SYNC)) {
      return (void *)fence;
   }

   return (void *)eglCreateSyncKHR(egl->egl_display, EGL_SYNC_FENCE_KHR, attrib_list);
}

void virgl_egl_wait_fence(struct virgl_egl *egl, void* sync)
{
   EGLSyncKHR fence = (EGLSyncKHR) sync;
   if (fence == EGL_NO_SYNC_KHR)
      return;
   eglWaitSyncKHR(egl->egl_display, fence, 0);
   eglDestroySyncKHR(egl->egl_display, fence);
}
