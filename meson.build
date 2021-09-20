#############################################################################
#
# Copyright (C) 2019 Collabora Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#

project(
   'virglrenderer', 'c',
   version: '0.8.2',
   license : 'MIT',
   meson_version : '>= 0.46',
   default_options : ['buildtype=release', 'b_ndebug=if-release',
                      'warning_level=3', 'c_std=gnu11']
)

# To change only before doing a release:
#
# 1. Incrememnt the revision
# 2. If the interface was changed in an compatible way increment the
#    interface age
# 3. If the ABI has changed in an incompatible way increment the binary_age
#    and set revision and interface_age to zero

binary_age    = 1
interface_age = 4
revision      = 2

cc = meson.get_compiler('c')

add_project_arguments('-DHAVE_CONFIG_H=1', language : 'c')
add_project_arguments('-D_GNU_SOURCE=1', language : 'c')

warnings = [
   '-Werror=implicit-function-declaration',
   '-Werror=missing-prototypes',
   '-Wmissing-prototypes',
   '-Werror=int-to-pointer-cast'
]

foreach w : warnings
   if cc.has_argument(w)
      add_project_arguments(w, language : 'c')
   endif
endforeach

flags = [
   '-fvisibility=hidden',
]

foreach f : flags
   if cc.has_argument(f)
      add_project_arguments(f, language : 'c')
   endif
endforeach

prog_python = import('python').find_installation('python3')

libdrm_dep = dependency('libdrm', version : '>=2.4.50', required: false)
thread_dep = dependency('threads')
epoxy_dep = dependency('epoxy', version: '>= 1.5.4')
m_dep = cc.find_library('m')

conf_data = configuration_data()
conf_data.set('VERSION', '0.8.1')

if cc.has_header('sys/uio.h')
   conf_data.set('HAVE_SYS_UIO_H', 1)
endif

if cc.has_header('dlfcn.h')
   conf_data.set('HAVE_DLFCN_H', 1)
endif

if cc.has_header('pthread.h')
   conf_data.set('HAVE_PTHREAD', 1)
endif

if cc.has_header('sys/eventfd.h')
   conf_data.set('HAVE_EVENTFD_H', 1)
endif

if cc.has_header('sys/select.h')
  conf_data.set('HAVE_SYS_SELECT_H', 1)
endif

if get_option('buildtype') == 'debug'
   add_global_arguments('-DDEBUG=1', language : 'c')
endif

if libdrm_dep.found()
  conf_data.set('HAVE_LIBDRM', 1)
endif

platforms = get_option('platforms')

require_egl = platforms.contains('egl')
require_glx = platforms.contains('glx')
auto_egl_glx = platforms.contains('auto')

with_egl = require_egl or auto_egl_glx
with_glx = require_glx or auto_egl_glx

have_egl = false
have_glx = false

egl_without_gbm = get_option('egl_without_gbm')

with_gbm_allocation = get_option('gbm_allocation')
if with_gbm_allocation
   if egl_without_gbm
      error('Cannot build gbm allocation without gbm')
   endif
   conf_data.set('ENABLE_GBM_ALLOCATION', 1)
   _gbm_ver = '18.0.0'
else
   _gbm_ver = '0.0.0'
endif

if with_egl
   if cc.has_header('epoxy/egl.h', dependencies: epoxy_dep) and epoxy_dep.get_pkgconfig_variable('epoxy_has_egl') == '1'
      if egl_without_gbm
         gbm_dep = dependency('', required : false)
         have_egl = true
         conf_data.set('EGL_WITHOUT_GBM', 1)
      else
         gbm_dep = dependency('gbm', version: '>= ' + _gbm_ver, required: require_egl)
         have_egl = gbm_dep.found()
      endif
      conf_data.set('HAVE_EPOXY_EGL_H', 1)
   else
      assert(not require_egl,
             'egl was explicitely requested but it is not supported by epoxy')
   endif
endif

if with_glx
   if cc.has_header('epoxy/glx.h', dependencies: epoxy_dep) and epoxy_dep.get_pkgconfig_variable('epoxy_has_glx') == '1'
      glx_dep = dependency('x11', required: require_glx)
      have_glx = glx_dep.found()
      conf_data.set('HAVE_EPOXY_GLX_H', 1)
   else
      assert(not require_glx,
             'glx was explicitely requested but it is not supported by epoxy')
   endif
endif

if cc.compiles('void __attribute__((hidden)) func() {}')
   conf_data.set('HAVE_FUNC_ATTRIBUTE_VISIBILITY', 1)
endif

configure_file(input : 'config.h.meson',
               output : 'config.h',
               configuration : conf_data)

pkgconf_data = configuration_data()
pkgconf_data.set('PACKAGE_VERSION', meson.project_version())
pkgconf_data.set('prefix', get_option('prefix'))
pkgconf_data.set('exec_prefix', '${prefix}')
pkgconf_data.set('libdir', '${prefix}/' + get_option('libdir'))
pkgconf_data.set('includedir', '${prefix}/' + get_option('includedir'))

pkg_config = configure_file(input : 'virglrenderer.pc.in',
                            output : 'virglrenderer.pc',
                            configuration : pkgconf_data)

install_data(pkg_config,
             install_dir: get_option('libdir') + '/pkgconfig')

inc_configuration = include_directories('.')

with_fuzzer = get_option('fuzzer')
with_tests = get_option('tests')
with_valgrind = get_option('valgrind')

subdir('src')
subdir('vtest')

if with_tests
   subdir('tests')
endif

lines = [
   '',
   'prefix:    ' + get_option('prefix'),
   'libdir:    ' + get_option('libdir'),
   '',
   'c_args:    ' + (' ').join(get_option('c_args')),
   '',
]

lines += 'egl:       ' + (have_egl ? 'yes' : 'no') + (egl_without_gbm ? ' (no gbm)' : '')
lines += 'glx:       ' + (have_glx ? 'yes' : 'no')
lines += ''
lines += 'gbm_alloc: ' + (with_gbm_allocation ? 'yes' : 'no' )
lines += ''
lines += 'tests:     ' + (with_tests ? 'yes' : 'no' )
lines += 'fuzzer:    ' + (with_fuzzer ? 'yes' : 'no' )

indent = '   '
summary = indent + ('\n' + indent).join(lines)
message('\n\nConfiguration summary:\n@0@\n'.format(summary))
