/**************************************************************************
 *
 * Copyright (C) 2015 Red Hat Inc.
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
#include <stdio.h>
#include <signal.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <fcntl.h>
#include <getopt.h>
#include <string.h>

#include "util.h"
#include "util/u_memory.h"
#include "vtest.h"
#include "vtest_protocol.h"
#include "virglrenderer.h"
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif


struct vtest_program
{
   const char *socket_name;
   int socket;
   const char *read_file;
   int in_fd;
   int out_fd;
   struct vtest_input input;

   const char *render_device;

   bool do_fork;
   bool loop;

   bool use_glx;
   bool use_egl_surfaceless;
   bool use_gles;
};

struct vtest_program prog = {
   .socket_name = VTEST_DEFAULT_SOCKET_NAME,
   .socket = -1,

   .read_file = NULL,

   .in_fd = -1,
   .out_fd = -1,
   .input = { { -1 }, NULL },
   .render_device = 0,
   .do_fork = true,
   .loop = true,
};

static void vtest_main_getenv(void);
static void vtest_main_parse_args(int argc, char **argv);
static void vtest_main_set_signal_child(void);
static void vtest_main_set_signal_segv(void);
static void vtest_main_open_read_file(void);
static void vtest_main_open_socket(void);
static void vtest_main_run_renderer(int in_fd, int out_fd, struct vtest_input *input,
                                    int ctx_flags, const char *render_device);
static void vtest_main_wait_for_socket_accept(void);
static void vtest_main_tidy_fds(void);
static void vtest_main_close_socket(void);

VIRGL_EXPORT
int main(int argc, char **argv)
{
#ifdef __AFL_LOOP
while (__AFL_LOOP(1000)) {
#endif

   vtest_main_getenv();
   vtest_main_parse_args(argc, argv);

   int ctx_flags = VIRGL_RENDERER_USE_EGL;
   if (prog.use_glx) {
      if (prog.use_egl_surfaceless || prog.use_gles) {
         fprintf(stderr, "Cannot use surfaceless or GLES with GLX.\n");
         exit(EXIT_FAILURE);
      }
      ctx_flags = VIRGL_RENDERER_USE_GLX;
   } else {
      if (prog.use_egl_surfaceless)
         ctx_flags |= VIRGL_RENDERER_USE_SURFACELESS;
      if (prog.use_gles)
         ctx_flags |= VIRGL_RENDERER_USE_GLES;
   }

   if (prog.read_file != NULL) {
      vtest_main_open_read_file();
      goto start;
   }

   if (prog.do_fork) {
      vtest_main_set_signal_child();
   }

   vtest_main_open_socket();
restart:
   vtest_main_wait_for_socket_accept();

start:
   if (prog.do_fork) {
      /* fork a renderer process */
      if (fork() == 0) {
         vtest_main_set_signal_segv();
         vtest_main_run_renderer(prog.in_fd, prog.out_fd, &prog.input,
                                 ctx_flags, prog.render_device);
         exit(0);
      }
   } else {
      vtest_main_set_signal_segv();
       vtest_main_run_renderer(prog.in_fd, prog.out_fd, &prog.input,
                               ctx_flags, prog.render_device);
   }

   vtest_main_tidy_fds();

   if (prog.loop) {
      goto restart;
   }

   vtest_main_close_socket();

#ifdef __AFL_LOOP
}
#endif
}

#define OPT_NO_FORK 'f'
#define OPT_NO_LOOP_OR_FORK 'l'
#define OPT_USE_GLX 'x'
#define OPT_USE_EGL_SURFACELESS 's'
#define OPT_USE_GLES 'e'
#define OPT_RENDERNODE 'r'
#define OPT_SOCKET_NAME 'n'

static void vtest_main_parse_args(int argc, char **argv)
{
   int ret;

   static struct option long_options[] = {
      {"no-fork",             no_argument, NULL, OPT_NO_FORK},
      {"no-loop-or-fork",     no_argument, NULL, OPT_NO_LOOP_OR_FORK},
      {"use-glx",             no_argument, NULL, OPT_USE_GLX},
      {"use-egl-surfaceless", no_argument, NULL, OPT_USE_EGL_SURFACELESS},
      {"use-gles",            no_argument, NULL, OPT_USE_GLES},
      {"rendernode",          required_argument, NULL, OPT_RENDERNODE},
      {"socket-name",         required_argument, NULL, OPT_SOCKET_NAME},
      {0, 0, 0, 0}
   };

   /* getopt_long stores the option index here. */
   int option_index = 0;

   do {
      ret = getopt_long(argc, argv, "", long_options, &option_index);

      switch (ret) {
      case -1:
         break;
      case OPT_NO_FORK:
         prog.do_fork = false;
         break;
      case OPT_NO_LOOP_OR_FORK:
         prog.do_fork = false;
         prog.loop = false;
         break;
      case OPT_USE_GLX:
         prog.use_glx = true;
         break;
      case OPT_USE_EGL_SURFACELESS:
         prog.use_egl_surfaceless = true;
         break;
      case OPT_USE_GLES:
         prog.use_gles = true;
         break;
      case OPT_RENDERNODE:
         prog.render_device = optarg;
         break;
      case OPT_SOCKET_NAME:
         prog.socket_name = optarg;
         break;
      default:
         printf("Usage: %s [--no-fork] [--no-loop-or-fork] [--use-glx] "
                "[--use-egl-surfaceless] [--use-gles] [--rendernode <dev>]"
                " [--socket-name <path>] [file]\n", argv[0]);
         exit(EXIT_FAILURE);
         break;
      }

   } while (ret >= 0);

   if (optind < argc) {
      prog.read_file = argv[optind];
      prog.loop = false;
      prog.do_fork = false;
   }
}

static void vtest_main_getenv(void)
{
   const char* socket_name = NULL;
   prog.use_glx = getenv("VTEST_USE_GLX") != NULL;
   prog.use_egl_surfaceless = getenv("VTEST_USE_EGL_SURFACELESS") != NULL;
   prog.use_gles = getenv("VTEST_USE_GLES") != NULL;
   prog.render_device = getenv("VTEST_RENDERNODE");
   socket_name = getenv("VTEST_SOCKET_NAME");
   if (socket_name != NULL) {
      prog.socket_name = socket_name;
   }
}

static void handler(int sig, siginfo_t *si, void *unused)
{
   (void)sig; (void)si, (void)unused;

   printf("SIGSEGV!\n");
   exit(EXIT_FAILURE);
}

static void vtest_main_set_signal_child(void)
{
   struct sigaction sa;
   int ret;

   memset(&sa, 0, sizeof(sa));
   sigemptyset(&sa.sa_mask);
   sa.sa_handler = SIG_IGN;
   sa.sa_flags = 0;

   ret = sigaction(SIGCHLD, &sa, NULL);
   if (ret == -1) {
      perror("Failed to set SIGCHLD");
      exit(1);
   }
}

static void vtest_main_set_signal_segv(void)
{
   struct sigaction sa;
   int ret;

   memset(&sa, 0, sizeof(sa));
   sigemptyset(&sa.sa_mask);
   sa.sa_flags = SA_SIGINFO;
   sa.sa_sigaction = handler;

   ret = sigaction(SIGSEGV, &sa, NULL);
   if (ret == -1) {
      perror("Failed to set SIGSEGV");
      exit(1);
   }
}

static void vtest_main_open_read_file(void)
{
   int ret;

   ret = open(prog.read_file, O_RDONLY);
   if (ret == -1) {
      perror(NULL);
      exit(1);
   }
   prog.in_fd = ret;
   prog.input.data.fd = prog.in_fd;
   prog.input.read = vtest_block_read;

   ret = open("/dev/null", O_WRONLY);
   if (ret == -1) {
      perror(NULL);
      exit(1);
   }
   prog.out_fd = ret;
}

static void vtest_main_open_socket(void)
{
   struct sockaddr_un un;

   prog.socket = socket(PF_UNIX, SOCK_STREAM, 0);
   if (prog.socket < 0) {
      goto err;
   }

   memset(&un, 0, sizeof(un));
   un.sun_family = AF_UNIX;

   snprintf(un.sun_path, sizeof(un.sun_path), "%s", prog.socket_name);

   unlink(un.sun_path);

   if (bind(prog.socket, (struct sockaddr *)&un, sizeof(un)) < 0) {
      goto err;
   }

   if (listen(prog.socket, 1) < 0){
      goto err;
   }

   return;

err:
   perror("Failed to setup socket.");
   exit(1);
}

static void vtest_main_wait_for_socket_accept(void)
{
   fd_set read_fds;
   int new_fd;
   int ret;
   FD_ZERO(&read_fds);
   FD_SET(prog.socket, &read_fds);

   ret = select(prog.socket + 1, &read_fds, NULL, NULL, NULL);
   if (ret < 0) {
      perror("Failed to select on socket!");
      exit(1);
   }

   if (!FD_ISSET(prog.socket, &read_fds)) {
      perror("Odd state in fd_set.");
      exit(1);
   }

   new_fd = accept(prog.socket, NULL, NULL);
   if (new_fd < 0) {
      perror("Failed to accept socket.");
      exit(1);
   }

   prog.in_fd = new_fd;
   prog.out_fd = new_fd;
   prog.input.data.fd = prog.in_fd;
   prog.input.read = vtest_block_read;
}

typedef int (*vtest_cmd_fptr_t)(uint32_t);

static const vtest_cmd_fptr_t vtest_commands[] = {
   NULL /* CMD ids starts at 1 */,
   vtest_send_caps,
   vtest_create_resource,
   vtest_resource_unref,
   vtest_transfer_get,
   vtest_transfer_put,
   vtest_submit_cmd,
   vtest_resource_busy_wait,
   NULL, /* vtest_create_renderer is a specific case */
   vtest_send_caps2,
   vtest_ping_protocol_version,
   vtest_protocol_version,
   vtest_create_resource2,
   vtest_transfer_get2,
   vtest_transfer_put2,
};

static void vtest_main_run_renderer(int in_fd, int out_fd,
                                    struct vtest_input *input, int ctx_flags,
                                    const char *render_device)
{
   int err, ret;
   uint32_t header[VTEST_HDR_SIZE];
   int initialized = 0;

   do {
      ret = vtest_wait_for_fd_read(in_fd);
      if (ret < 0) {
         err = 1;
         break;
      }

      ret = input->read(input, &header, sizeof(header));
      if (ret < 0 || (size_t)ret < sizeof(header)) {
         err = 2;
         break;
      }

      if (!initialized) {
         /* The first command MUST be VCMD_CREATE_RENDERER */
         if (header[1] != VCMD_CREATE_RENDERER) {
            err = 3;
            break;
         }

         ret = vtest_create_renderer(input, out_fd, header[0], ctx_flags, render_device);
         if (ret < 0) {
            err = 4;
            break;
         }
         initialized = 1;
         printf("%s: vtest initialized.\n", __func__);
         vtest_poll();
         continue;
      }

      vtest_poll();
      if (header[1] <= 0 || header[1] >= ARRAY_SIZE(vtest_commands)) {
         err = 5;
         break;
      }

      if (vtest_commands[header[1]] == NULL) {
         err = 6;
         break;
      }

      ret = vtest_commands[header[1]](header[0]);
      if (ret < 0) {
         err = 7;
         break;
      }

      /* GL draws are fenced, while possible fence creations are too */
      if (header[1] == VCMD_SUBMIT_CMD || header[1] == VCMD_RESOURCE_CREATE ||
          header[1] == VCMD_RESOURCE_CREATE2)
         vtest_renderer_create_fence();

   } while (1);

   fprintf(stderr, "socket failed (%d) - closing renderer\n", err);

   vtest_destroy_renderer();
}

static void vtest_main_tidy_fds(void)
{
   // out_fd will be closed by the in_fd clause if they are the same.
   if (prog.out_fd == prog.in_fd) {
      prog.out_fd = -1;
   }

   if (prog.in_fd != -1) {
      close(prog.in_fd);
      prog.in_fd = -1;
      prog.input.read = NULL;
   }

   if (prog.out_fd != -1) {
      close(prog.out_fd);
      prog.out_fd = -1;
   }
}

static void vtest_main_close_socket(void)
{
   if (prog.socket != -1) {
      close(prog.socket);
      prog.socket = -1;
   }
}
