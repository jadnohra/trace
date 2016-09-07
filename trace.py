from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from ctypes import *
from collections import namedtuple
import inspect
import math
import socket
import shlex
import struct
import time
import fnmatch
import copy
import csv
import shutil
import json


try:
	import numpy
except ImportError:
    numpy = None

g_defaults = { 'g_proc':'n{}'.format(32), 's_proc':'n{}'.format(1), 'path':'', 'sample_dt':0.0, 'dbg_conv':False }


def m_min(v1, v2):
	return v1 if (v1 <= v2) else v2
def m_max(v1, v2):
	return v1 if (v1 >= v2) else v2
def m_interp(v1, v2, t):
	return v1*(1.0-t)+v2*(t)
def m_abs(v):
	return v if (v >= 0) else -v
def v2_z():
	return [0.0, 0.0]
def v2_abs(v):
	return [m_abs(v[0]), m_abs(v[1])]
def v2_add(v1, v2):
	return [v1[0]+v2[0], v1[1]+v2[1]]
def v2_sub(v1, v2):
	return [v1[0]-v2[0], v1[1]-v2[1]]
def v2_dot(v1, v2):
	return v1[0]*v2[0]+v1[1]*v2[1]
def v2_lenSq(v1):
	return v2_dot(v1, v1)
def v2_len(v1):
	return math.sqrt(v2_lenSq(v1))
def v2_distSq(p1, p2):
	vec = v2_sub(p2, p1)
	return v2_lenSq(vec)
def v2_dist(p1, p2):
	vec = v2_sub(p2, p1)
	return v2_len(vec)
def v2_muls(v1, s):
	return [v1[0]*s, v1[1]*s]
def v3_interp(v0, v1, t):
	return [m_interp(v0[0], v1[0], t), m_interp(v0[1], v1[1], t), m_interp(v0[2], v1[2], t)]

xstr = lambda s: str(s) if (s is not None) else ''
xcol = lambda x,y,z: [float(x)/255.0,float(y)/255.0,float(z)/255.0]
aget = lambda a,i,d: a[i] if len(a) > i else d

def data_adjust_bounds(d, removed):
	if (d['miny'] is not None) and d['is_ordered']:
		if (d['miny'] in removed or d['maxy'] in removed):
			d['miny'] = None

def create_dprocessor(name, process):
	c = { 'name':name, 'process':process }
	return c

def create_dprocessor_pass(name):
	def func(p, d):
		pass
	p = create_dprocessor(name, func)
	return p

def create_dprocessor_add(name):
	def func(p, d):
		d['pts'] = [sum([float(x) for x in d['pts']])]
	p = create_dprocessor(name, func)
	return p

def create_dprocessor_rle(name):
	def func(p, d):
		pts = d['pts']
		if (len(pts)):
			last = not pts[-1]
			n = 0
			for p in reversed(pts):
				if (p == last):
					n = n+1
				else:
					break
			if (n > 0):
				data_adjust_bounds(d, d['pts'][-n:])
				d['pts'] = pts[:-n]
	p = create_dprocessor(name, func)
	return p

def create_dprocessor_len(name):
	def func(p, d):
		npts = d['len']
		if (len(d['pts']) > npts):
			#two modes, overflow mode and slow mode, recover slow mode, use -test
			data_adjust_bounds(d, d['pts'][:-npts])
			d['pts'] = d['pts'][-npts:]
	c = create_dprocessor(name, func)
	return c

def create_dprocessor_period(name):
	def func(p, d):
		period = d['len']; l = len(d['pts'])
		if (l > period):
			left = max(((l/period)*period)-l, 1)
			data_adjust_bounds(d, d['pts'][:-left])
			d['pts'] = d['pts'][-left:]
	c = create_dprocessor(name, func)
	return c

def create_processor_len(name, n):
	def func(p, d):
		npts = c['len']
		if (len(d['pts']) > npts):
			#two modes, overflow mode and slow mode, recover slow mode, use -test
			data_adjust_bounds(d, d['pts'][:-npts])
			d['pts'] = d['pts'][-npts:]
	c = create_dprocessor(name, func)
	c['len'] = n
	return c

def create_processor_period(name, n):
	def func(p, d):
		period = c['len']; l = len(d['pts'])
		if (l > period):
			left = max(((l/period)*period)-l, 1)
			data_adjust_bounds(d, d['pts'][:-left])
			d['pts'] = d['pts'][-left:]
	c = create_dprocessor(name, func)
	c['len'] = n
	return c

g_dprocessor_pass = create_dprocessor_pass('pass')
g_dprocessor_add = create_dprocessor_add('add')
g_dprocessor_rle = create_dprocessor_rle('rle')
g_dprocessor_len = create_dprocessor_len('npts')
g_dprocessor_period = create_dprocessor_period('period')

d_numeric_types = ['f', 'hf', 'i', 'vf', 'vhf', 'vhd']
d_string_types = ['s']
d_types = d_numeric_types + d_string_types
d_filters = []

def data_is_numeric(type):
	return type in d_numeric_types
def data_is_vector(type):
	return type.startswith('v')
def data_is_ordered(type):
	return data_is_numeric(type) and (not data_is_vector(type))

def to_data_processors(pstr):
	err_ret = None
	if (pstr.startswith('n')):
		try:
			n = int(pstr[1:])
			proc = g_dprocessor_len
			return (n, [proc])
		except Exception as e:
			print str(e); return err_ret;
	elif (pstr.startswith('p')):
		try:
			n = int(pstr[1:]);
			proc = g_dprocessor_period
			return (n, [proc])
		except Exception as e:
			print str(e); return err_ret;
	elif (pstr == '+'):
		return (2, [g_dprocessor_add])
	return err_ret

def to_processors(pstr):
	err_ret = None
	if (pstr.startswith('n')):
		try:
			n = int(pstr[1:])
			proc = create_processor_len(None, n)
			return (n, [proc])
		except Exception as e:
			print str(e); return err_ret;
	elif (pstr.startswith('p')):
		try:
			n = int(pstr[1:]);
			proc = create_processor_period(None, n)
			return (n, [proc])
		except Exception as e:
			print str(e); return err_ret;
	elif (pstr == '+'):
		return (2, [g_dprocessor_add])
	return err_ret

def set_data_proc(data, procstr):
	procs = to_data_processors(procstr)
	if procs is None:
		return False
	data['len'] = procs[0]; data['processors'] = procs[1];
	return True

def create_data(index, name, type, procstr, orig_info):
	d = { 'name':name, 'in':0, 'pts':[], 'add':[], 'blocked':False, 'update':None, 'len':None, 'processors':None, 'orig_info':orig_info,
			'type':type, 'is_numeric': data_is_numeric(type), 'is_vector': data_is_vector(type), 'is_ordered': data_is_ordered(type),
	 		'lminy':None, 'lmaxy':None, 'miny':None, 'maxy':None,
			'bind':None, 'last_bind':None, 'bind_name':None,
			'index':index, 'sample_dt':0.0 }
	if (set_data_proc(d, procstr) == False):
		set_data_proc(d, 'n3')
	if (len(d_filters)):
		allowed = False
		for f in d_filters:
			if f in name:
				allowed = True
		d['data']['blocked'] = (allowed is False)
	return d

def set_data(datas, name, type, procstr, orig_info):
	if (name not in datas):
		datas[name] = create_data(len(datas), name, type, procstr, orig_info)
	return datas[name]

def bind_data(data, rep):
	b = { 'data':data, 'rep':rep }
	rep['bind_func'](b)
	data['last_bind'] = data['bind']
	data['bind'] = b
	data['bind_name'] = rep['name']

def unbind_data(data, rep):
	data['last_bind'] = data['bind']
	data['bind'] = None
	data['bind_name'] = None

def add_data_pt(data, y):
	if (data['blocked'] is False):
		if (data['type'] == 'hf'):
			y = h2f(y)
		elif (data['type'] == 'i'):
			y = int(y)
		elif (data['type'] == 'f'):
			y = float(y)
		elif (data['type'] == 'vf'):
			y = [float(x) for x in y]
		elif (data['type'] == 's'):
			y = str(y)
		elif (data['type'] == 'hd'):
			y = h2d(y)
		elif (data['type'] == 'vhf'):
			y = [h2f(x)	for x in y.split(',')]
		elif (data['type'] == 'vhd'):
			y = [h2d(x)	for x in y.split(',')]
		else:
			pass
		data['add'].append(y)
		if (data['is_ordered']):
			lminy = data['lminy']; lminy = y if (lminy is None) else lminy;
			lmaxy = data['lmaxy']; lmaxy = y if (lmaxy is None) else lmaxy;
			data['lminy'] = m_min(lminy, y)
			data['lmaxy'] = m_max(lmaxy, y)
			miny = data['miny']; miny = y if (miny is None) else miny;
			maxy = data['maxy']; maxy = y if (maxy is None) else maxy;
			data['miny'] = m_min(miny, y)
			data['maxy'] = m_max(maxy, y)

def process_data_pts(d, reps):
	if (d['blocked'] is False):
		if d['update']:
			d['update'](d)
		d['in'] = d['in']+len(d['add'])
		d['pts'].extend(d['add'])
		for p in d['processors']:
			p['process'](p,d)
		d['add'] = []
		if (d['bind'] is not None):
			procs = d['bind']['rep']['processors']
			for p in procs:
				p['process'](p,d)
		if (d['is_ordered'] and (d['miny'] is None) and (len(d['pts']))):
			d['miny'] = min(d['pts']); d['maxy'] = max(d['pts']);

def data_life_index(d, i):
	return d['in']-(len(d['pts'])-i)

g_rtypes = ['g', 's']
g_types = ['y', 'xy']

def set_graph_type(g, type):
	if (g['type'] != type):
		if (type == 'xy'):
			g['processors'] = [g_dprocessor_rle]
		else:
			g['processors'] = []
		g['type'] = type

def create_graph(name, type, fit, miny, maxy, lines):
	def bind(b):
		b.update({ 'hidden':False, 'focus':False, 'select':False, 'fit': fit, 'fit_life': True })
	g = { 'rtype': 'g', 'name':name, 'type':None, 'lines':lines, 'miny':miny, 'maxy':maxy,
			'bind_func':bind, 'window':None, 'processors':None, 'fold':True }
	set_graph_type(g, type)
	return g

def set_graph(graphs, name, type, fit, miny, maxy, lines):
	if (name not in graphs):
		graphs[name] = create_graph(name, type, fit, miny, maxy, lines)
	return graphs[name]

def get_graph_info(g,d,b):
	deco = '';
	deco = deco if not b['hidden'] else deco+'[h]'
	deco = deco if not d['blocked'] else deco+'[b]'
	deco = deco if not d['name'] == g_exclusive else deco+'[e]'
	deco = deco if not b['select'] else deco+'[s]'
	deco = deco if not b['fit'] else deco+'[F]'
	deco = deco if not b['fit_life'] else deco+'[L]'
	#return '({})<{}>{} [{}, {}] [{}, {}] [{}, {}]'.format(g['index'], g['name'], deco, xstr(g['miny']), xstr(g['maxy']), xstr(g['fminy']), xstr(g['fmaxy']), xstr(g['lminy']), xstr(g['lmaxy']))
	return '({})<{}>{} [{}, {}]'.format(d['index'], d['name'], deco, xstr(g['miny']), xstr(g['maxy']))

def create_stat(name, procs):
	def bind(b):
		pass
	s = { 'rtype':'s', 'name':name, 'processors':procs[1], 'bind_func':bind }
	return s

def set_stat(stats, name, procstr = 'n1'):
	if (name not in stats):
		stats[name] = create_stat(name, to_processors(procstr))
		stats[name]['procstr'] = procstr
	return stats[name]

def make_stat_string(s, d):
	return '{}: {}'.format(d['name'], d['pts'])

def get_stat_info(s, d):
	return '({})<{}:{}>'.format(d['index'], d['name'], d['type'])

Bound = namedtuple('Bound', 'min max')

def make_bound(pt1, pt2):
	return Bound( [m_min(pt1[0], pt2[0]), m_min(pt1[1], pt2[1])], [m_max(pt1[0], pt2[0]), m_max(pt1[1], pt2[1])] )

def clip_to_bound(pt, bound):
	return [ m_min(m_max(pt[0], bound.min[0]), bound.max[0]), m_min(m_max(pt[1], bound.min[1]), bound.max[1]) ]

def is_in_bound(pt, bound):
	return clip_to_bound(pt, bound) == pt

def merge_bounds(bd1, bd2):
	return Bound( [m_min(bd1[0][0], bd2[0][0]), m_min(bd1[0][1], bd2[0][1])], [m_max(bd1[1][0], bd2[1][0]), m_max(bd1[1][1], bd2[1][1])] )

def inc_bound(bd, inc):
	return Bound( v2_sub(bd[0], inc), v2_add(bd[1], inc) )

def scale_bound(bd, fac):
	inc = v2_sub(bd[1], bd[0]); v2_muls(inc, (fac-1.0)*0.5)
	return inc_bound(bd, inc)


def style0_color3f(r,g,b):
	return (r, g, b)

def style1_color3f(r,g,b):
	return (1.0-r, 1.0-g, 1.0-b)

style_color3f = style0_color3f

def style_glColor3f(r,g,b):
	glColor3f(*style_color3f(r,g,b))

def draw_bound(bound, mode):
	glBegin(mode)
	glVertex2f(bound.min[0],bound.min[1])
	glVertex2f(bound.min[0],bound.max[1])
	glVertex2f(bound.max[0],bound.max[1])
	glVertex2f(bound.max[0],bound.min[1])
	glVertex2f(bound.min[0],bound.min[1])
	glEnd()

def trace_bound(bound, col):
	style_glColor3f(col[0],col[1],col[2])
	draw_bound(bound, GL_LINE_STRIP)

def fill_bound(bound, col):
	style_glColor3f(col[0],col[1],col[2])
	draw_bound(bound, GL_POLYGON)

def screen_to_draw(pt, wh):
	x,y,z = gluUnProject(pt[0], wh-pt[1], 0);
	return [x,y];

def size_to_draw(sz, wh):
	p0 = screen_to_draw(v2_z(), wh); p = screen_to_draw(sz, wh); return v2_sub(p, p0);

def get_string_height(): return 13
def get_string_size(str): return [len(str)*8, get_string_height()]

def draw_strings(strs, x0, y0, col, wh, anchor = 'lt', fill_col = None):

	if (anchor in ['cc', 'ct', 'lb']):
		bounds = [0, 0]
		for str in strs:
			sz = get_string_size(str)
			bounds[0] = m_max(bounds[0], sz[0]); bounds[1] = bounds[1] + sz[1]
		dbounds = size_to_draw(bounds, wh)
		if (anchor[0] == 'c'):
			x0 = x0 - 0.5*dbounds[0]
		if (anchor[1] == 'c'):
			y0 = y0 - 0.5*dbounds[1]
		elif (anchor[1] == 'b'):
			y0 = y0 - dbounds[1]

	bounds = []
	if (True) and (fill_col is not None) and len(strs):
		for str in strs:
			style_glColor3f(col[0],col[1],col[2])
			h = size_to_draw([0.0, get_string_height()], wh)[1]
			si = 0
			for str in strs:
				sz = size_to_draw(get_string_size(str), wh)
				bounds.append(Bound([x0,(y0+h)+(si*h)], [x0+sz[0], (y0+h)+((si-1)*h)]))
				si = si+1

		bound = bounds[0];
		for b in bounds:
			bound = merge_bounds(bound, b)

		inc = v2_abs(size_to_draw([3.0, 5.0], g_wind_ext[1]))
		fill_bound(inc_bound(bound, inc), fill_col)

	calc_bounds = (len(bounds) == 0)
	style_glColor3f(col[0],col[1],col[2])
	h = size_to_draw([0.0, get_string_height()], wh)[1]
	glPushMatrix();
	glTranslatef(x0, y0+h, 0); glRasterPos2f(0, 0);
	si = 0
	for str in strs:
		for c in str:
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))
		sz = size_to_draw(get_string_size(str), wh)
		if (calc_bounds):
			bounds.append(Bound([x0,(y0+h)+(si*h)], [x0+sz[0], (y0+h)+((si-1)*h)]))
		glTranslatef(0, h, 0); glRasterPos2f(0, 0);
		si = si+1
	glPopMatrix()

	return bounds

g_deferred_draw_strings = []

def defer_draw_strings(strs, x0, y0, col, wh, anchor = 'lt', fill_col = None):
	global g_deferred_draw_strings
	g_deferred_draw_strings.append([strs, x0, y0, col, wh, anchor, fill_col])

def flush_defer_draw_strings():
	global g_deferred_draw_strings
	for args in g_deferred_draw_strings:
		draw_strings(*args)
	g_deferred_draw_strings = []


ClosestPt = namedtuple('ClosestPt', 'mpt dist pt y i extra')

def draw_closest_pt(g, d, b, closest):
	closest_str = None
	if (closest and (closest.dist is not None)):
		b['closest_pt'] = [data_life_index(d, closest.i), closest.y]
		glPointSize(6.0)
		glBegin(GL_POINTS)
		style_glColor3f(1.0,0.5,0.0)
		glVertex2f(closest.pt[0],closest.pt[1])
		glEnd()
		glBegin(GL_LINE_STRIP)
		glVertex2f(closest.mpt[0],closest.mpt[1])
		glVertex2f(closest.pt[0],closest.pt[1])
		glEnd()
	else:
		if ('closest_pt' in b):
			del b['closest_pt']

def draw_graph(g, d, b, bound, track, mpt, col, fast_mode, freq_mode, analyse_conv):
	if (g['type'] == 'y'):
		draw_graph_y(g, d, b, bound, track, mpt, col, fast_mode, freq_mode, analyse_conv)
	elif (g['type'] == 'xy'):
		draw_graph_xy(g, d, b, bound, track, mpt, col, fast_mode)

g_vbo_context = []
def vbo_request(key):
	if (len(g_vbo_context) == 0):
		vbo = glGenBuffers(1)
		g_vbo_context.append(vbo)
	return g_vbo_context[0]

def vbo_release(key, vbo):
	pass

def vbo_beginverts():
	glEnableClientState(GL_VERTEX_ARRAY); #glEnableClientState(GL_COLOR_ARRAY);
	vbo = vbo_request(None)
	glBindBuffer(GL_ARRAY_BUFFER, vbo)
	return vbo

def vbo_endverts(vbo, verts, mode):
	glBufferData (GL_ARRAY_BUFFER, len(verts)*4, (c_float*len(verts))(*verts), GL_STREAM_DRAW)
	glVertexPointer(2, GL_FLOAT, 0, None)
	glDrawArrays(mode, 0, len(verts)/2)
	glDisableClientState(GL_VERTEX_ARRAY); #glDisableClientState(GL_COLOR_ARRAY);
	vbo_release(None, vbo)

def draw_graph_begin_func_direct(col, ptsz, use_lines):
	style_glColor3f(col[0], col[1], col[2]); glPointSize(ptsz);
	if use_lines:
		glBegin(GL_LINE_STRIP)
	else:
		glBegin(GL_POINTS)
	return use_lines

def draw_graph_end_func_direct(ctx):
	glEnd()

def draw_graph_vert_func_direct(ctx, pt, col):
	style_glColor3f(col[0],col[1],col[2])
	glVertex2f(pt[0],pt[1])

def draw_graph_begin_func_vbo(col, ptsz, use_lines):
	style_glColor3f(col[0], col[1], col[2]); glPointSize(ptsz);
	return (vbo_beginverts(), use_lines, [])

def draw_graph_end_func_vbo(ctx):
	vbo_endverts(ctx[0], ctx[2], GL_LINE_STRIP if ctx[1] else GL_POINTS)

def draw_graph_vert_func_vbo(ctx, pt, col):
	ctx[2].extend(pt); #ctx[3].extend(col);

def xstr(string):
	return string if (string is not None) else ''

def unit_conv_id(y):
	return [y, '']

def unit_conv_rad_deg(y):
	return [(y*180.0)/(math.pi), ' deg.']

def unit_conv_m_cm(y):
	return [(y*10.0), ' cm.']

g_unit_conv_funcs = { 'rad-deg':unit_conv_rad_deg, 'm-cm':unit_conv_m_cm, 'id':unit_conv_id }
g_track_unit_conv = 'id'

def draw_strings_graph_y(g,d,b, bound, info_strings, status_string, closest):
	info_strings = []
	if (closest.dist is not None):
		y_conv = g_unit_conv_funcs.get(g_track_unit_conv, 'id')(closest.y)
		closest_str = '{}. {} {}'.format(data_life_index(d, closest.i), '{}{}'.format(y_conv[0], y_conv[1]), xstr(closest.extra))
		info_strings.append(closest_str)
	draw_strings(info_strings, bound.min[0] + size_to_draw([2.0, 0.0], g_wind_ext[1])[0], bound.max[1], [1.0]*3, g_wind_ext[1])

	status_string = '[{}] {}'.format(d['name'], status_string)
	left_bottom = screen_to_draw([3.0, g_wind_ext[1] - 3.0], g_wind_ext[1])
	draw_strings([status_string], left_bottom[0], left_bottom[1], [1.0]*3, g_wind_ext[1], 'lb')


PatternEl = namedtuple('PatternEl', 'kind conv pert range avg')
g_pattern = { 'pattern':[], 'hit':0, 'buff':[], 'len':16 }

def draw_graph_y_impl(g,d,b, pts, npts, sample_dt, range_min, range_max, use_lines, bound, track, mpt, col, enable_analyse_conv, begin_func, vert_func, end_func):

	info_strings = []
	status_string = ''

	min_dist = None; min_y = None; min_pt = None; min_i = None;
	acol1 = [0.3*col[0],0.3*col[1],0.3*col[2]]; acol2 = [0.5,0.3,0.3];
	bcol1 = [col[0],col[1],col[2]]; bcol2 = [1.0,0.0,0.0];
	xdiv = (bound.max[0]-bound.min[0]) / float(npts)
	yscl0 = 1.0 / (range_max-range_min)
	yscl1 = (bound.max[1]-bound.min[1])
	split_i = npts
	is_split = len(pts) > npts
	col1 = bcol1; col2 = bcol2;
	if (is_split):
		col1 = acol1; col2 = acol2;

	def pt_to_draw(pt):
		x = bound.min[0]+pt[0]*xdiv
		t = (pt[1]-range_min) * yscl0
		y = bound.min[1]+ t * yscl1
		out = [x,y]
		cout = clip_to_bound(out, bound)
		return (cout, out)

	if (enable_analyse_conv):

		Kind_Unsure = 0; Kind_Mon = 1; Kind_Oscill = 2; Kind_Bump = 3;
		Conv_Unsure = 4; Conv_Const = 5; Conv_Conv = 6; Conv_Div = 7; Conv_PertConv = 8; Conv_PertDiv = 9;
		Wave_Short = 10; Res_Yes = 11; Res_No = 12;
		WaveString = ['unsure', 'monotone', 'oscill', 'bump', 'unsure', 'const', 'conv', 'div', 'p-conv', 'p-div', 'short', 'yes', 'no']

		def extract_local_extrema(w):
			n = len(w); out = []; li = 0;
			while (li < n):
				mi = li+1
				while (mi+1 < n and w[mi] == w[mi+1]):
					mi = mi+1
				ri = mi+1
				while (ri+1 < n and w[ri] == w[ri+1]):
					ri = ri+1
				if (ri < n):
					lmax = w[li] < w[mi] and w[mi] > w[ri]
					lmin = w[li] > w[mi] and w[mi] < w[ri]
					if (lmax or lmin):
						last_is_lmax = lmax
						out.append(mi)
				li = mi
			return out

		def extract_envelope(w, extrema, sgn):
			if (len(extrema) >= 3):
				gmax1 = [-1, -1, 0.0]; gmax2 = [-1, -1, 0.0];
				for ei in range(0 if sgn*w[extrema[0]] > sgn*w[extrema[1]] else 1, len(extrema), 2):
					if (gmax1[0] == -1 or sgn*w[extrema[ei]] > sgn*gmax1[2]):
						if (gmax2[0] != -1):
							gmax2[0] = -1 # only look forward from a global maximum, since we are interested in conv. in the future.
							#gmax2 = gmax1
						gmax1 = [extrema[ei], ei, w[extrema[ei]]]
					if (ei != gmax1[1]):
						if (gmax2[0] == -1 or sgn*w[extrema[ei]] > sgn*gmax2[2]):
							gmax2 = [extrema[ei], ei, w[extrema[ei]]]
				supp1 = gmax1; supp2 = gmax2;
				if (supp2[0] != -1):

					def take_slope(bi, a, dir, sgn):
						slopei = (bi[1]-a[1]) / (bi[0]-a[0])
						take_slopei = (sgn*slopei > sgn*slope) if (dir > 0) else (sgn*slope > sgn*slopei)
						return (take_slopei, slopei)

					a = [float(supp1[0]), w[supp1[0]]]; b0 = [float(supp2[0]), w[supp2[0]]];
					slope = (b0[1]-a[1]) / (b0[0]-a[0])
					dir = 2 if (supp2[1] > supp1[1]) else -2
					for ti in range(supp2[1]+dir, len(extrema) if dir > 0 else 0, dir):
						bi = [float(extrema[ti]), w[extrema[ti]]]
						(take_slopei, slopei) = take_slope(bi, a, dir, sgn)
						if (take_slopei):
							supp2 = [extrema[ti], ti, w[extrema[ti]]]
							slope = slopei
					wi = len(w)-1 if dir > 0 else 0
					bi = [float(wi), w[wi]]
					(take_slopei, slopei) = take_slope(bi, a, dir, sgn)
					if (take_slopei):
						supp2 = [wi, -1, w[wi]]
				return (supp1[0], supp2[0])
			return (-1, -1)

		def extract_envelopes(w, extrema):
			out = []
			for ienv in [-1.0, 1.0]:
				(env1_i, env2_i) = extract_envelope(pts, extrema, ienv)
				out.append((env1_i, env2_i))
			return out

		def calc_envelope_points(pts, envelope):
			if (envelope[0] != -1 and envelope[1] != -1):
				a = [float(envelope[0]), pts[envelope[0]]]; b = [float(envelope[1]), pts[envelope[1]]];
				return (a, b)
			else:
				return None

		def intersect_lines(p1, p2, p3, p4):
			det = (p4[1]-p3[1])*(p2[0]-p1[0]) - (p4[0]-p3[0])*(p2[1]-p1[1])
			if (det == 0):
				return None
			ua = ((p4[0]-p3[0])*(p1[1]-p3[1]) - (p4[1]-p3[1])*(p1[0]-p3[0])) / det
			return v2_add(p1, v2_muls(v2_sub(p2, p1), ua))

		def analyse_envelope_conv(lenv_pts, uenv_pts, n, dt):
			out = {}
			if (not lenv_pts or not uenv_pts):
				out = {'decision':'unsure'}
			else:
				inters_pt = intersect_lines(lenv_pts[0], lenv_pts[1], uenv_pts[0], uenv_pts[1])
				if (inters_pt is None):
					out = {'decision':'oscill'}
				else:
					if (inters_pt[0] < 0.0):
						out = { 'decision':'div' }
					else:
						out = { 'decision':'conv', 'conv':inters_pt, 'time':(inters_pt[0]-n)*dt }

				amax = m_max(uenv_pts[0][1], uenv_pts[1][1])
				amin = m_max(lenv_pts[0][1], lenv_pts[1][1])
				range = [amin, amax]
				out['range'] = range

			return out

		def zero_cross_tangent(a, b, n, dt):
			inters_pt = intersect_lines(a, b, [0,0], [1,0])
			return (inters_pt, ((inters_pt[0]-n)*dt) if inters_pt else float('inf') )

		def time_slope(a, b, dt):
			if (a[1] == b[1]):
				return 0.0
			return (b[1]-a[1]) / (dt*(b[0]-a[0]))

		def predict_tangent_drift(a, b, n, dt, time):
			slope = time_slope(a, b, dt)
			drift_y = a[1]+time*slope
			return (drift_y - b[1])

		def analyse_monotone_conv_sgn(w, sgn, optimistic):
			if (optimistic):
				if (len(w) == 0):
					return (Wave_Short, 0.0)
			else:
			 	if (len(w) < 2):
					return (Wave_Short, 0.0)
			pert = 0.0
			n = len(w)
			for i in range(1, len(w)):
				diff = w[i]-w[i-1]
				if (i > 1):
					#wrong! must use slope, a line of ct slope does not converge!
					d2 = sgn*m_abs(diff); d1 = sgn*m_abs(last_diff);
					if (d2 > d1):
						pert += d2-d1
					#elif (d2 == d1):
					#	pert += d2-d1
				last_diff = diff
			pert = pert / float(n)
			return (Res_Yes if (pert == 0.0) else Res_No, pert)

		def analyse_monotone_conv(w, optimistic = False):
			(conv, pconv) = analyse_monotone_conv_sgn(w, 1, optimistic)
			(div, pdiv) = analyse_monotone_conv_sgn(w, -1, optimistic)

			if (conv == Wave_Short or div == Wave_Short):
				return (Wave_Short, 0.0)

			if (conv == Res_Yes or div == Res_Yes and conv != div):
				ret_conv = Conv_Const if (conv == Res_Yes and div == Res_Yes) else (Conv_Conv if conv == Res_Yes else Conv_Div)
				return (ret_conv, 0.0)
			else:
				if (pconv < pdiv):
					return (Conv_PertConv, pconv)
				else:
					return (Conv_PertDiv, pdiv)

		def split_extrema(w, extrema):
			first_max = 0 if w[extrema[0]] > w[0] else 1
			first_min = 1-first_max
			maxima = [extrema[xi] for xi in range(first_max, len(extrema), 2)]
			minima = [extrema[xi] for xi in range(first_min, len(extrema), 2)]
			return (minima, maxima)

		def analyse_oscill_conv(w, extrema):

			def analyse_oscill_conv_compl(w, wi, compl_wi):
				kind = Kind_Unsure; conv = Conv_Unsure; pert = 0.0;
				if (len(wi)):
					kind = Kind_Oscill
					wv = [w[x] for x in wi]
					(conv, pert) = analyse_monotone_conv(wv, True)
				else:
					if (len(compl_wi)):
						kind = Kind_Bump
						mon_start = compl_wi[-1]+1
						(conv, pert) = analyse_monotone_conv(w[mon_start:], False)
					else:
						kind = Wave_Short
				return (kind, conv, pert)

			if (len(extrema) == 0):
				ret_short = (Wave_Short, Conv_Unsure, 0.0)
				return ( ret_short, ret_short, ret_short)
			(minima, maxima) = split_extrema(w, extrema)

			(max_kind, max_conv, max_pert) = analyse_oscill_conv_compl(w, maxima, minima)
			(min_kind, min_conv, min_pert) = analyse_oscill_conv_compl(w, minima, maxima)

			conv = Conv_Unsure; pert = 0.0;
			if ((max_conv == Conv_Conv or max_conv == Conv_Const)
					and (min_conv == Conv_Conv or min_conv == Conv_Const)):
				conv = Conv_Conv
			elif (max_conv == Conv_Div or min_conv == Conv_Div):
				conv = Conv_Div
			elif ((max_conv == Conv_Conv or max_conv == Conv_Const or max_conv == Conv_PertConv)
					and (min_conv == Conv_Conv or min_conv == Conv_Const or min_conv == Conv_PertConv)):
				conv = Conv_PertConv; pert = m_max(max_pert, min_pert);
			elif ((max_conv == Conv_Div or max_conv == Conv_PertDiv)
					or (min_conv == Conv_Div or min_conv == Conv_PertDiv)):
				conv = Conv_PertDiv; pert = m_max(max_pert, min_pert);

			kind = Kind_Oscill
			if (max_kind == Kind_Bump or min_kind == Kind_Bump):
				kind = Kind_Bump

			return ( (kind, conv, pert), (max_kind, max_conv, max_pert), (min_kind, min_conv, min_pert) )

		def extract_zero_cross(w):
			out = []
			for i in range(1, len(w)):
				if (w[i-1]*w[i] <= 0):
					a = [float(i-1), w[i-1]]; b = [float(i), w[i]];
					d = v2_sub(b, a)
					if (d[1] != 0):
						t = -a[1]/d[1]
						zc = v2_add(a, v2_muls(d, t))
					else:
						zc = [a[0], 0.0]
					out.append(zc)
			return out

		extrema = extract_local_extrema(pts)

		def analyse_conv(w, extrema = None, info_strs = None):
			(mon_conv, mon_pert) = analyse_monotone_conv(w)
			if (mon_conv != Conv_Unsure and mon_pert == 0.0):
				return (Kind_Mon, mon_conv, mon_pert)
			extrema = extrema if extrema else extract_local_extrema(w)
			oscill_kinds = analyse_oscill_conv(w, extrema)
			if (info_strs is not None):
				info_strs.append(str([(WaveString[x[0]], WaveString[x[1]], x[2]) for x in oscill_kinds]))
			return (oscill_kinds[0][0], oscill_kinds[0][1], oscill_kinds[0][2])

		if (enable_analyse_conv and False): # == 1):

			(wave_kind, wave_conv, wave_pert) = analyse_conv(pts, extrema)
			pert_str = '({0:.4g}) '.format(wave_pert) if wave_pert else ''
			range_val = max(pts) - min(pts)
			range_str = '[{0:.4g}]'.format(range_val)
			avg_val = sum(pts)/float(len(pts))
			avg_str = '~{0:.4g}'.format(avg_val)
			info_str = '{}, {} {}{} {}'.format(WaveString[wave_kind], WaveString[wave_conv], pert_str, range_str, avg_str)
			info_strings.append(info_str)

			if 0:
				def comp_pel(pel1, pel2, eps = 0.01):
					return ((pel1.kind == pel2.kind) and (pel1.conv == pel2.conv)
							#and (m_abs(pel1.pert-pel2.pert) < eps)
							#and (m_abs(pel1.range-pel2.range) < eps)
							#and (m_abs(pel1.avg-pel2.avg) < eps)
							)

				pel = PatternEl(WaveString[wave_kind], WaveString[wave_conv], wave_pert, range_val, avg_val)
				pbuff = g_pattern['buff']

				if (len(pbuff) == 0 or comp_pel(pel, pbuff[-1]) == False):
					if (len(g_pattern['buff']) >= g_pattern['len']):
						pbuff.pop(0)
					pbuff.append(pel)
					found_pattern = False
					for i in range(len(pbuff)):
						for j in range(i+1, len(pbuff)):
							if (comp_pel(pbuff[i], pbuff[j])):
								diff = j-i
								if (j+diff < len(pbuff)):
									match = True
									for dd in range(1, diff):
										if (comp_pel(pbuff[i+dd], pbuff[j+dd]) == False):
											match = False
									if (match):
										found_pattern = True
										part = pbuff[i:i+diff]; pattern_part = g_pattern['pattern'];
										patt_match = False
										if (len(part) == len(pattern_part)):
											patt_match = True
											for k in range(len(part)):
												if (comp_pel(part[k], pattern_part[k]) == False):
													patt_match = False
										if (patt_match):
											g_pattern['hit'] = g_pattern['hit']+1
										else:
											g_pattern['pattern'] = part
											g_pattern['hit'] = 0
										g_pattern['buff'] = g_pattern['buff'][i+diff:]

					if (found_pattern == False and (len(g_pattern['buff']) >= g_pattern['len'])):
						g_pattern['pattern'] = []; g_pattern['hit'] = 0;

				if (g_pattern['hit']):
					info_strings.append('pattern[{}] x {}'.format(len(g_pattern['pattern']), g_pattern['hit']))

		# elif (enable_analyse_conv == 2):

		# 	def make_range_signal(w, rle = True):
		# 		if (len(w) == 0):
		# 			return []
		# 		rw = []
		# 		min = w[0]; max = w[0]; r = max-min; rw.append(r)
		# 		for v in w:
		# 			min = m_min(min, w); max = m_max(max, w)
		# 			r = max-min
		# 			if (rle) or (r != rw[-1]):
		# 				rw.append(r)
		# 		return rw

		# 	def make_avg_signal(w, rle = True):
		# 		if (len(w) == 0):
		# 			return []
		# 		aw = []; caw = 0.0;
		# 		for i in range(len(w)):
		# 			caw = caw + (v-caw)/(float)(i+1)
		# 			if (len(aw) == 0) or (rle) or (caw != aw[-1]):
		# 				aw.append(caw)
		# 		return aw

		# 	rw = make_range_signal(w, False)
		# 	if (len(rw) >= 3):
		# 		needed_samples = 3; counts = {};
		# 		(last_conv, last_pert) = analyse_monotone_conv(rw[:3], False)
		# 		count[last_conv] = count.get(last_conv, 0)+1
		# 		for ni in range(4, len(rw)):
		# 			(conv, pert) = analyse_monotone_conv(rw[:ni], False)
		# 			if (conv != last_conv):
		# 				last_conv = conv; last_pert = pert;
		# 				count[last_conv] = count.get(last_conv, 0)+1
		# 				needed_samples = ni

		# 		if (count.get(Conv_Div, 0) + ount.get(Conv_PertDiv, 0) == 0):
		# 		else:

		else:
			ends_range = [m_min(pts[0], pts[-1]), m_max(pts[0], pts[-1])]
			if (len(extrema)):
				pts_range = [min([pts[x] for x in extrema]), max([pts[x] for x in extrema])]
				pts_range = [m_min(ends_range[0], pts_range[0]), m_max(ends_range[1], pts_range[1])]
			else:
				pts_range = ends_range
			info_str = '|{:.2e}, {:.2e}| = {:.2e}'.format(pts_range[0], pts_range[1], pts_range[1]-pts_range[0])
			status_string = status_string + info_str

		glPointSize(4.0); style_glColor3f(*xcol(230,90,70)); glBegin(GL_POINTS);
		for ei in extrema:
			a = [float(ei), pts[ei]];
			(cex, ex) = pt_to_draw(a)
			glVertex2f(*cex)
		glEnd()

		if (False): # Envelopes
			envelopes = extract_envelopes(pts, extrema)
			lenv_pts = calc_envelope_points(pts, envelopes[0])
			uenv_pts = calc_envelope_points(pts, envelopes[1])
			both_env_pts = [lenv_pts, uenv_pts]
			for env_pts in both_env_pts:
				if (env_pts):
					(ca, uca) = pt_to_draw(env_pts[0]); (cb, ucb) = pt_to_draw(env_pts[1]);
					glBegin(GL_LINES); style_glColor3f(*xcol(85,127,255)); glVertex2f(*ca);  glVertex2f(*cb); glEnd();
					glPointSize(8.0); glBegin(GL_POINTS); style_glColor3f(*xcol(255,0,0)); glVertex2f(*ca); glEnd();

			conv_anal = analyse_envelope_conv(lenv_pts, uenv_pts, len(pts), sample_dt)
			if (g_defaults['dbg_conv']):
				print conv_anal
			if (conv_anal['decision'] == 'conv'):
				convy = conv_anal['conv'][1]
				(ca, uca) = pt_to_draw([0, convy]); (cb, ucb) = pt_to_draw([len(pts), convy]);
				glBegin(GL_LINES); style_glColor3f(*xcol(165,165,165)); glVertex2f(*ca);  glVertex2f(*cb); glEnd();

		zero_cross = extract_zero_cross(pts)
		glPointSize(4.0); style_glColor3f(*xcol(255,170,85)); glBegin(GL_POINTS);
		for zc in zero_cross:
			(czc, zc) = pt_to_draw(zc)
			glVertex2f(*czc)
		glEnd()


	ctx = begin_func(col1, 1.25, use_lines)
	off = 0
	for i in range(len(pts)):
		if (i == split_i):
			off = split_i
			col1 = bcol1; col2 = bcol2;
			end_func(ctx); ctx = begin_func(col1, 1.25, use_lines);
		j = i - off
		(cpt, pt) = pt_to_draw([float(j), pts[i]])
		col = col1 if (pt == cpt) else col2
		if (track):
			dist = v2_dist(cpt, mpt)
			if (min_dist is None or min_dist > dist):
				min_dist = dist; min_pt = cpt; min_y = pts[i]; min_i = i;
		vert_func(ctx, cpt, col)
	end_func(ctx)

	if (track):
		closest = ClosestPt(mpt, min_dist, min_pt, min_y, min_i, None)
		draw_closest_pt(g,d,b, closest)
		draw_strings_graph_y(g,d,b, bound, info_strings, status_string, closest)


def draw_graph_y_freq_impl(g,d,b, pts, npts, sample_dt, range_min, range_max, use_lines, bound, track, mpt, col, analyse_conv, begin_func, vert_func, end_func):

	N = len(pts)
	def_sampling = g_defaults['sample_dt']
	T = sample_dt if (sample_dt > 0.0) else (def_sampling if (def_sampling > 0.0) else (1.0/N))
	#x = np.linspace(0.0, N*T, N)
	yf = numpy.fft.fft(pts)
	yf = [2.0*numpy.abs(x) for x in yf[0:N/2]]
	#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
	#plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
	fpts = yf
	fpts.pop(0)

	if (len(fpts) == 0):
		return

	range_max = 1.1*max(fpts); range_min = 0.0;

	def xaxis(x):
		#return x
		return 10.0*math.log10(float(x+1))

	min_dist = None; min_y = None; min_pt = None; min_i = None; min_freq = None;
	col1 = [col[0],col[1],col[2]]; col2 = [1.0,0.0,0.0];
	xdiv = (bound.max[0]-bound.min[0]) / xaxis(float(npts))
	yscl0 = 1.0 / (range_max-range_min)
	yscl1 = (bound.max[1]-bound.min[1])

	ctx = begin_func(col1, 1.25, use_lines)
	for i in range(len(fpts)):
		j = i
		x = bound.min[0]+xaxis(float(j))*xdiv
		t = (fpts[i]-range_min) * yscl0
		y = bound.min[1]+ t * yscl1
		pt = [x,y]
		cpt = clip_to_bound(pt, bound)
		col = col1 if (pt == cpt) else col2
		if (track):
			dist = v2_dist(cpt, mpt)
			if (min_dist == None or min_dist > dist):
				min_dist = dist; min_pt = cpt; min_y = fpts[i]; min_i = i; min_freq = float(i+1)/(T*N); #0 hz was removed
		vert_func(ctx, cpt, col)
	end_func(ctx)

	if (track):
		closest = ClosestPt(mpt, min_dist, min_pt, min_y, min_i, '@ {:.2} Hz.'.format(min_freq))
		draw_closest_pt(g,d,b, closest)

		info_strings = []; status_string = '';
		draw_strings_graph_y(g,d,b, bound, info_strings, status_string, closest)


def draw_graph_y(g, d, b, bound, track, mpt, col, fast_mode, freq_mode, enable_analyse_conv):
	if (len(d['pts']) == 0):
		return

	pts = d['pts']
	npts = d['len']
	if (enable_analyse_conv == False) and (g['fold'] == True):
		npts = m_max(1, d['len']/2)

	range_min = g['miny']; range_max = g['maxy'];
	if b['fit']:
		if (b['fit_life']):
			range_min = d['lminy']; range_max = d['lmaxy'];
		else:
			range_min = d['miny']; range_max = d['maxy'];
		over = float((npts-m_min(npts,(len(pts))))) / float(m_max(npts, 1))
		if (range_min == range_max):
			range_max = range_min + 0.1
		diff = 	range_max-range_min
		range_min = range_min - diff * ((1.5 * over) + (0.2 * (1.0-over)))
		range_max = range_max + diff * ((1.5 * over) + (0.2 * (1.0-over)))

	if fast_mode:
		begin_func = draw_graph_begin_func_vbo; vert_func = draw_graph_vert_func_vbo; end_func = draw_graph_end_func_vbo;
	else:
		begin_func = draw_graph_begin_func_direct; vert_func = draw_graph_vert_func_direct; end_func = draw_graph_end_func_direct;
	draw_impl = draw_graph_y_freq_impl if (freq_mode and d['is_ordered']) else draw_graph_y_impl

	sample_dt = d['sample_dt']
	if enable_analyse_conv:
		sample_dt = (sample_dt if sample_dt > 0.0 else 1.0/60.0)
	closest = draw_impl(g,d,b, pts, npts, sample_dt, range_min, range_max, g['lines'], bound, track, mpt, col, enable_analyse_conv, begin_func, vert_func, end_func)
	draw_closest_pt(g,d,b, closest)

def draw_graph_xy_impl(pts, range_min, range_max, use_lines, bound, track, mpt, col, begin_func, vert_func, end_func):
	min_dist = None; min_y = None; min_pt = None; min_i = None;
	scol = [col[0]*0.01,col[1]*0.01,col[2]*0.01]; ecol = [col[0],col[1],col[2]]; col2 = [1.0,0.0,0.0];
	ctx = begin_func(ecol, 1.25, use_lines)
	scl0 = 1.0 / (range_max-range_min); sclx1 = (bound.max[0]-bound.min[0]); scly1 = (bound.max[1]-bound.min[1]);
	ti = 0.0; cdiv = 1.0/float(len(pts));
	for i in range(len(pts)):
		tx = (pts[i][0]-range_min) * scl0; x = bound.min[0]+ tx * sclx1;
		ty = (pts[i][1]-range_min) * scl0; y = bound.min[1]+ ty * scly1;
		pt = [x,y]
		cpt = clip_to_bound(pt, bound)
		col1 = v3_interp(scol, ecol, ti*cdiv)
		ti = ti + 1.0
		col = col1 if (pt == cpt) else col2
		if (track):
			dist = v2_dist(cpt, mpt)
			if (min_dist == None or min_dist > dist):
				min_dist = dist; min_pt = cpt; min_y = pts[i]; min_i = i;
		vert_func(ctx, cpt, col)
	end_func(ctx)
	return ClosestPt(mpt, min_dist, min_pt, min_y, min_i, None)

def draw_graph_xy(g, d, b, bound, track, mpt, col, fast_mode):
	if (len(d['pts']) == 0):
		return

	pts = d['pts']
	npts = m_max(1, d['len']/2)

	range_min = g['miny']; range_max = g['maxy'];
	if b['fit']:
		if (b['fit_life']):
			range_min = d['lminy']; range_max = d['lmaxy'];
		else:
			range_min = d['miny']; range_max = d['maxy'];
		over = float((npts-m_min(npts,(len(pts))))) / float(m_max(npts, 1))
		if (range_min == range_max):
			range_max = range_min + 0.1
		diff = 	range_max-range_min
		range_min = range_min - diff * ((1.5 * over) + (0.2 * (1.0-over)))
		range_max = range_max + diff * ((1.5 * over) + (0.2 * (1.0-over)))

	if fast_mode:
		begin_func = draw_graph_begin_func_vbo; vert_func = draw_graph_vert_func_vbo; end_func = draw_graph_end_func_vbo;
	else:
		begin_func = draw_graph_begin_func_direct; vert_func = draw_graph_vert_func_direct; end_func = draw_graph_end_func_direct;
	draw_graph_xy_impl(pts, range_min, range_max, g['lines'], bound, track, mpt, col, begin_func, vert_func, end_func)


def create_bounds(n, w, h, woff, hoff):
	if n == 0:
		return []
	w = float(w); h = float(h)
	a = math.sqrt((w*h)/float(n))
	mul = 0.5; thresh = w*0.01; dir = 0.0;
	calc_n = lambda w,h,x: int(math.floor(h/x)*math.floor(w/x))
	eps = 1.0+thresh
	an = calc_n(w,h,a)
	while(eps > thresh or an < n):
		la = a
		if (an > n):
			if (dir == -1.0):
				mul = 0.5*mul
			dir = 1.0
			a = a + mul*a
		elif (an < n):
			if (dir == 1.0):
				mul = 0.5*mul
			dir = -1.0
			a = a - mul*a
		else:
			dir = 1.0
			a = a + mul*a
		an = calc_n(w,h,a)
		eps = math.fabs(a-la)

	ret = []
	x=0;y=0;nx=int(w/a);
	for i in range(n):
		if (x >= nx):
			x = 0; y = y+1;
		bound = Bound([woff+(x*a), hoff+(y*a)], [woff+((x+1.0)*a), hoff+((y+1.0)*a)])
		ret.append(bound)
		x = x+1
	return ret

def create_margins(bounds, margin):
	ret = []
	for b in bounds:
		m = Bound( v2_add(b.min, margin), v2_sub(b.max, margin) )
		ret.append(m)
	return ret

def h2f(h):
	return struct.unpack('!f', h.zfill(8).decode('hex'))[0]

def h2d(h):
	return struct.unpack('!d', h.zfill(8).decode('hex'))[0]

g_mouse = [None, None]
g_buttons = {}
g_keys = {}
g_special_keys = {}
g_track = True
g_mouseFocus = True

def handleKeys(key, x, y):
	global g_keys
	g_keys[key] = {'wpt':[x,y] }

def handleSpecialKeys(key, x, y):
	global g_special_keys
	g_special_keys[str(key)] = {'wpt':[x,y] }

def handleMouseAct(button, mode, x, y):
	global g_buttons
	g_buttons[button] = {'button': button, 'mode':mode, 'wpt':[x,y] }

def handleMousePassiveMove(x, y):
	global g_mouse
	if (g_mouseFocus):
		g_mouse = [x,y]

def handleMouseMove(x, y):
	handleMousePassiveMove(x, y)

def handleMouseEntry(state):
	global g_mouseFocus
	g_mouseFocus = (state == GLUT_ENTERED)

g_export_state = { 'sock':None, 'base_path':None, 'buff_len':128, 'recv_t0':0, 'needs_export':False, 'in_count':0, 'prog_in_count':0, 'prog_i':0,
					'progress':['\\', '|', '/', '-', '|', '/', '-']
				 }

def cmd_make_path(path):
	if (not os.path.exists(path)):
		os.makedirs(path)
	return os.path.exists(path)

def cmd_export_data(data, close):

	if (data['path'] is None):
		name_path = data['name'].replace('\\', '/')
		dir_count = name_path.count('/')
		name_path = data['name'].replace('/', '_', dir_count-1)
		name_path = '_'.join(name_path.split())
		data['path'] = os.path.join(g_export_state['base_path'], '{}.stxt'.format(name_path))

	if (data['fo'] is None) and (data['failed'] == False) and (data['path'] is not None):
		fo = None
		write_mode = 'w' if (data['exports'] == 0) else 'a'
		if (cmd_make_path(os.path.dirname(data['path']))):
			fo = open(data['path'], write_mode)
		if (fo):
			fo.write('# data\n')
			fo.write('{{ "name":"{}", "type":"{}" }}\n'.format(data['name'], data['type']))
			fo.write('# points\n')
		else:
			print 'Failed to create [{}]'.format(data['path'])
			data['Failed'] = True
		data['fo'] = fo

	if (data['fo'] is not None):
		fo = data['fo']
		for pt in data['pts']:
			fo.write(pt); fo.write('\n');
		fo.flush()
		data['exports'] = data['exports'] + 1
		if (close):
			fo.close(); data['fo'] = None;
	data['pts'] = []

def cmd_export_flush():
	global g_datas

	for data in g_datas.values():
		cmd_export_data(data, True)

def cmd_export_add_pt(datas, name, type, y):
	buff_len = g_export_state['buff_len']
	data = g_datas.get(name, None)
	if (data is None):
		data = { 'name':name, 'pts':[], 'type':type, 'path':None, 'fo':None, 'failed':False, 'exports':0 }
		g_datas[name] = data
	pts = data['pts']
	pts.append(y)
	if (len(pts) >= buff_len):
		cmd_export_data(data, False)

def find_indexed_dir(base_path, prefix, start_index):
	def is_empty_dir(path):
		try:
			return os.path.isdir(path) and len(os.listdir(path)) == 0
		except:
			path
		return False

	dirs = []; files = [];
	index = start_index; sess_dir = '{}{}'.format(prefix, index); found = -1;
	for x in os.listdir(base_path):
		if (x == sess_dir):
			xpath = os.path.join(base_path, x)
			if (is_empty_dir(xpath) == False):
				found = index; index = index+1; sess_dir = '{}{}'.format(prefix, index)
	return found

def make_auto_session_path(base_path):
	prefix = 's'
	index = find_indexed_dir(base_path, prefix, 1)
	index = m_max(1, index + 1)
	return os.path.join(base_path, '{}{}'.format(prefix, index))

def get_last_session_path(base_path):
	prefix = 's'
	index = find_indexed_dir(base_path, prefix, 1)
	return base_path if (index < 0) else os.path.join(base_path, '{}{}'.format(prefix, index))

def get_export_usage():
	return 'export mode: -export path [-clear] [-auto_sess] [-last_sess]'

def prepare_export(ip, port):

	base_path = sys_argv_get(['-export'], None)
	if (base_path is None):
		print get_export_usage();
		return False

	if (cmd_make_path(base_path) == False):
		print 'could not create path [{}]'.format(base_path)
		return False

	force_auto_sess = False
	if (sys_argv_has(['-last_sess'])):
		new_path = get_last_session_path(base_path)
		force_auto_sess = (new_path == base_path)
		base_path = new_path

	if (force_auto_sess or sys_argv_has(['-auto_sess'])):
		base_path = make_auto_session_path(base_path)
		if (cmd_make_path(base_path) == False):
			print 'could not create path [{}]'.format(base_path)
			return False

	if sys_argv_has(['-clear']):
		print 'clearing [{}]...'.format(base_path)
		shutil.rmtree(base_path)
		cmd_make_path(base_path)

	print 'exporting to [{}]...'.format(base_path)
	print 'quit with [Esc] or [q]'.format(base_path)
	g_export_state['base_path'] = base_path
	g_export_state['sock'] = bind_socket(ip, port)

	return True


def process_sock_export(console_print):
	global g_export_state

	sock = g_export_state['sock']
	sock_data = '0'
	while len(sock_data):
		try:
			sock_data,address = sock.recvfrom(128)
		except Exception:
			sock_data = ''

		if (sock_data):
			g_export_state['recv_t0'] = time.time(); g_export_state['needs_export'] = True; g_export_state['in_count'] = g_export_state['in_count']+1;
			if g_dbg:
				print sock_data
			token_list = shlex.split(sock_data)
			if (len(token_list) >= 4 and token_list[0] == 'pt'):
				#pt: name type y
				cmd_export_add_pt(g_datas, token_list[1], token_list[2], token_list[3])
			elif (len(token_list) >= 1 and token_list[0] == 'flush'):
				#flush
				cmd_export_flush()

			if console_print and ( (g_export_state['in_count'] == 1) or (g_export_state['in_count'] - g_export_state['prog_in_count'] > 1024) ):
				g_export_state['prog_in_count'] = g_export_state['in_count']; g_export_state['prog_i'] = g_export_state['prog_i']+1;
				#sys.stdout.flush()
				msg = g_export_state['progress'][g_export_state['prog_i']%len(g_export_state['progress'])]*8
				print '{}\r'.format(msg),

	if (g_export_state['needs_export']):
			dt = time.time() - g_export_state['recv_t0']
			if (dt > 1.5):
				g_export_state['needs_export'] = False
				cmd_export_flush()
				if (console_print):
					print '\nFlushed'
				else:
					print 'Export flushed'

def main_export():

	msvcrt = None

	try:
		import msvcrt
	except ImportError:
		print 'cannot poll keyboard, kill the process to stop.'

	def kb_poll():
		if msvcrt:
			return ord(msvcrt.getch()) if msvcrt.kbhit() else 0
		else:
			return 0

	ip = sys.argv[sys.argv.index('-ip')+1] if '-ip' in sys.argv else "127.0.0.1"
	port = int(sys.argv[sys.argv.index('-port')+1] if '-port' in sys.argv else "17641")

	if (prepare_export(ip, port) == False):
		return 0

	while True:
		process_sock_export(True)
		kb_hit = kb_poll()
		if (kb_hit):
			if (kb_hit == ord('\x1b') or kb_hit == ord('q')):
				if (g_export_state['needs_export']):
					cmd_export_flush()
					print '\nFlushed'
				return 0


g_gallery_state = { 'base_path':None, 'curr_path':None, 'curr_dirs':[], 'name_single':'_gal_0', 'names_vec3':['_gal_v3_0', '_gal_v3_1', '_gal_v3_2'], 'max_files':32, 'data_prefix':'_' }

def get_gallery_usage():
	return 'gallery mode: -gallery path [-max_files number] [-last_sess]'

def prepare_gallery():

	gal = g_gallery_state

	base_path = sys_argv_get(['-gallery'], None)
	if (base_path is None and sys_argv_has(['-gallery'])):
		print get_gallery_usage()
		return False

	if ( (base_path is not None) and (sys_argv_has(['-last_sess'])) ):
		base_path = get_last_session_path(base_path)

	gal['base_path'] = base_path
	gal['curr_path'] = base_path
	gal['max_files'] = int(sys_argv_get(['-max_files'], gal['max_files']))

	if (base_path is not None):
		g = set_graph(g_reps, gal['name_single'], 'y', True, -100.0, 100.0, True); g['fold'] = False;
		g = set_graph(g_reps, gal['names_vec3'][0], 'y', True, -100.0, 100.0, True); g['window'] = gal['names_vec3'][0]; g['fold'] = False;
		g = set_graph(g_reps, gal['names_vec3'][1], 'y', True, -100.0, 100.0, True); g['window'] = gal['names_vec3'][0]; g['fold'] = False;
		g = set_graph(g_reps, gal['names_vec3'][2], 'y', True, -100.0, 100.0, True); g['window'] = gal['names_vec3'][0]; g['fold'] = False;

	return True

def open_file_gallery(path):

	def new_datas(all_datas, index, name, type):
		gal = g_gallery_state; datas = []; is_vec = data_is_vector(type); prefix = gal['data_prefix'];
		if (is_vec):
			new_type = type.replace('v', '')
			for i in range(3):
				#r = g_reps[gal['names_vec3'][i]];
				r = g_reps[gal['name_single']]
				dname = '{}{}_v3_{}'.format(prefix, name, i)
				if (dname not in all_datas):
					d = create_data(data_index, dname, new_type, 'n512', 'g'); d['index'] = data_index; all_datas[dname] = d; bind_data(d, r);
				datas.append(all_datas[dname])
		else:
			r = g_reps[gal['name_single']]
			dname = '{}{}'.format(prefix, name)
			if (dname not in all_datas):
				d = create_data(data_index, dname, type, 'n512', 'g'); d['index'] = data_index; all_datas[dname] = d; bind_data(d, r);
			datas.append(all_datas[dname])
		return datas

	def finalize_block(datas):
		for d in datas:
			set_data_proc(d, 'n{}'.format(len(d['add'])))

	data_index = len(g_datas)
	all_datas = {}

	try :
		if 1:
			section = ''
			curr_datas = []
			with open(path, 'r') as fi:
				li = 0
				for line in fi:
					line = line.strip()
					if (len(line)):
						if (line.startswith('# data')):
							section = 'data'
						elif (line.startswith('# points')):
							section = 'points'
						else:
							if (section == 'data'):
								json_data = json.loads(line)
								name = json_data['name']; type = json_data['type'];
								datas = new_datas(all_datas, data_index, name, type)
								curr_datas = datas
							elif (section == 'points'):
								if (len(curr_datas)):
									token_list = [x.strip() for x in line.split(',')]
									for i in range(len(token_list)):
										y = token_list[i]
										add_data_pt(datas[i], y)
		if 0:
			curr_block_datas = []
			with open(path, 'r') as fi:
				fdata = fi.read()
				if len(fdata) >= 4:
					ending = fdata[-4:].strip().replace('\n', '').replace('\r', '')
					if (ending.endswith(']}') == False):
						fdata = fdata + ']}'
					json_glob = '{"frags":[' + fdata + ']}'
					print json.loads(json_glob)
					json_frags = fdata.strip().split('}')
					for json_frag in json_frags:
						json_frag = json_frag.strip()
						if (len(json_frag)):
							json_frag = json_frag + '}'
							json_data = json.loads(json_frag)
							name = json_data['name']; type = json_data['type']; points = json_data['points'];
							datas = new_block(name, type)
							if (datas):
								all_datas.extend(datas)
								curr_block_datas = datas
								if (len(curr_block_datas)):
									for pt in points:
										token_list = [x.strip() for x in pt.split(',')]
										for i in range(len(token_list)):
											y = token_list[i]
											add_data_pt(datas[i], y)
	except Exception:
		print traceback.format_exc()
		return None

	finalize_block(all_datas.values())
	return all_datas.values()


def close_files_gallery():
	global g_datas

	prefix = g_gallery_state['data_prefix']
	names = g_datas.keys()
	for name in names:
		if (name.startswith(prefix)):
			del g_datas[name]


g_sock = None
g_datas = {}
g_reps = {}
g_state = ''
g_exclusive = None
g_pause = False
g_frame = -1
g_dbg = False
g_freq_mode = False
g_fast_mode = False
g_analyse_conv = False
g_fps = 0
g_fps_t0 = 0
g_fps_frames =0
g_tab = ''
g_tabs = [g_tab]
g_autobinds = []
g_defaultbinds = []
g_cmd = ''
g_long_cmd = False
g_wind_ext = None
g_mouse_graph = False
g_graph_filtering = {'only':[], 'hide':[] }
g_client_fps = {}
g_cli_fps_dead_dt = 2*60.0

def graph_filter_allow(name):
	if (len(g_graph_filtering['only'])):
		for key in g_graph_filtering['only']:
			if key in name:
				return True
		return False
	if (len(g_graph_filtering['hide'])):
		for key in g_graph_filtering['hide']:
			if key in name:
				return False
		return True
	return True

def get_bound_reps(datas, tab):
	bgraphs = 	[ x['bind'] for x in datas.values() if (x['bind'] is not None) and (tab == x.get('tab', '')) and (x['bind']['rep']['rtype'] == 'g') ]
	bstats = 	[ x['bind'] for x in datas.values() if (x['bind'] is not None) and (tab == x.get('tab', '')) and (x['bind']['rep']['rtype'] == 's') ]
	return (bgraphs, bstats)

def add_tab(tab):
	global g_tabs
	if (not tab in g_tabs):
		g_tabs.append(tab)

def next_tab():
	global g_tabs
	global g_tab
	if (len(g_tabs) == 0):
		pass
	i = g_tabs.index(g_tab) if g_tab in g_tabs else 0
	i = (i+1)%len(g_tabs)
	g_tab = g_tabs[i]

def assign_tab(dname, tname):
	if (dname in g_datas):
		g_datas[dname]['tab'] = tname

def cmd_add_pt(datas, reps, name, type, procstr, y, bind, orig_info):
	data = g_datas.get(name, None)
	if (data is None):
		data = set_data(g_datas, name, type, procstr, orig_info)
		#data['sample_dt'] = 1.0/60.0
		if (bind is not None) and (data['bind_name'] != bind):
			r = reps.get(bind, None)
			if r:
				bind_data(data, r)
	add_data_pt(data, y)

def match_datas(data_dict, name_filter, rep_filter, do_print = True):
	rtype = None if (rep_filter is None) else ('s' if rep_filter.startswith('s') else 'g')
	filt_match = lambda d: (rtype is None) or ((d['bind'] != None) and (d['bind']['rep']['rtype']==rtype))
	dnames = [x for x in data_dict.keys() if x in data_dict and fnmatch.fnmatch(x, name_filter) and filt_match(data_dict[x])]
	if (do_print and len(dnames) > 1 and len(name_filter) and name_filter != '*'):
		print 'matched:', dnames
	return [data_dict[x] for x in dnames]

def cmd_mul_len(datas, factor, min):
	max = 0
	for d in datas:
		d['len'] = m_max(min, int(d['len']*factor))
		max = m_max(d['len'], max)
	return max

def cmd_export_datas(datas, path, dft_path):
	if (len(dft_path) > 0 and os.path.isabs(path) == False):
		path = os.path.join(dft_path, path)
	fname, fext = os.path.splitext(path)
	if (len(fext) == 0):
		if (os.path.isdir(path) == False):
			os.makedirs(path)
	for d in datas:
		fp = os.path.join(path, d['name']) if (len(fext) == 0) else path
		with open(fp, 'w') as fo:
			pts = None
			if (d['is_vector']):
				pts = [str(v) for v in d['pts']]
			else:
				pts = [str(x) for x in d['pts']]
			fo.write(', '.join(pts))


def cmd_set_default_sample_dt(sampling):
	smpl = float(sampling)
	g_defaults['sample_dt'] = smpl if (smpl > 0.0 and smpl != float('inf')) else 0.0

def cmd_set_default_hz(hz):
	hz = float(hz)
	cmd_set_default_sample_dt(1.0/hz if hz > 0.0 else 0.0)

def create_cli_fps(rep, name, t1):
	global g_client_fps
	global g_datas
	r = g_reps['_default_s' if rep == 's_fps' else '_default_g']; d = set_data(g_datas, 'cli_fps_{}'.format(name), 'i', 'n64', 's'); bind_data(d, r);
	g_client_fps[name] = { 't0': t1, 'frames': 0, 'fps':0, 'd':d, 'life_t':t1 }

def update_cli_fps(name, t1):
	cli_fps = g_client_fps[name]
	cli_fps['life_t'] = t1
	cli_fps['frames'] = cli_fps['frames']+1
	if t1 - cli_fps['t0'] >= 1000.0:
		cli_fps['fps'] = (1000.0 * float(cli_fps['frames'])) / float(t1-cli_fps['t0'])
		cli_fps['t0'] = t1; cli_fps['frames'] = 0
		add_data_pt(cli_fps['d'], int(cli_fps['fps']) )

mouse_graph_name = '<mouse>'
def cmd_mouse_graph(enable):
	global g_reps, g_datas
	def update_input(d):
		if g_mouse and g_mouse[1] and g_wind_ext and g_mouse_graph:
			mouse_pt = screen_to_draw(g_mouse, g_wind_ext[1])
			add_data_pt(d, mouse_pt[1])
	name = mouse_graph_name
	if (enable):
		r = set_graph(g_reps, name, 'y', 1, -2.0, 2.0, True);
		d = set_data(g_datas, name, 'f', 'n1024', 'g');
		d['update'] = update_input; bind_data(d, r);
		r['col'] = xcol(255, 255, 85)
		#d['index'] = -1
	#else:
	#	del g_reps[name]; del g_datas[name];

g_last_prnt_from_addr = None

def process_sock_realtime():
	global g_sock
	global g_last_prnt_from_addr

	if g_sock:
		t1 = glutGet(GLUT_ELAPSED_TIME)
		data = '0'
		while len(data):
			try:
				data,address = g_sock.recvfrom(128)
			except Exception:
				data = ''

			if (data and not g_pause):
				token_list = shlex.split(data)
				if (token_list[0] == 'echo'):
					g_sock.sendto(data[len('echo '):], address)
					print 'echoing from {}:{}: [{}]'.format(address[0], address[1], data)
				elif (token_list[0] == 'prnt'):
					if (g_last_prnt_from_addr != address):
						print 'print from {}:{}'.format(address[0], address[1])
						g_last_prnt_from_addr = address
					print data[len('prnt '):]
				else:
					if g_dbg:
						print data
				if (len(token_list) >= 4 and token_list[0] == 'pt'):
					#spt: name type y [bind]
					cmd_add_pt(g_datas, g_reps, token_list[1], token_list[2], g_defaults['g_proc'], token_list[3], aget(token_list, 4, '_default_g'), 'g')
				elif (len(token_list) >= 4 and token_list[0] == 's_pt'):
					#spt: name type y [bind]
					cmd_add_pt(g_datas, g_reps, token_list[1], token_list[2], g_defaults['s_proc'], token_list[3], aget(token_list, 4, '_default_s'), 's')
				elif (len(token_list) >= 4 and token_list[0] == '+_pt'):
					#spt: name type y [bind]
					cmd_add_pt(g_datas, g_reps, token_list[1], token_list[2], '+', token_list[3], aget(token_list, 4, '_default_s'), 's')
				elif (len(token_list) >= 5 and token_list[0] == 'xpt'):
					#pt: name type proc y [bind]
					cmd_add_pt(g_datas, g_reps, token_list[1], token_list[2], token_list[3], token_list[4], aget(token_list, 5, '_default_g'), 'g')
				elif (len(token_list) >= 5 and token_list[0] == 's_xpt'):
					#pt: name type proc y [bind]
					cmd_add_pt(g_datas, g_reps, token_list[1], token_list[2], token_list[3], token_list[4], aget(token_list, 5, '_default_s'), 's')
				elif (len(token_list) == 7 and token_list[0] == 'graph'):
					#print token_list
					# graph: graphs, name, type, fit, miny, maxy, lines
					set_graph(g_reps, token_list[1], token_list[2], int(token_list[3]), h2f(token_list[4]), h2f(token_list[5]), int(token_list[6]))
				elif (len(token_list) == 3 and token_list[0] == 'stat'):
					# stat: name, type
					#print token_list
					set_stat(g_reps, token_list[1], token_list[2])
				elif (len(token_list) == 2 and token_list[0] == 'hz'):
					# hz: hz.
					cmd_set_default_hz(float(token_list[1]))
				elif (token_list[0] == 's_fps' or token_list[0] == 'g_fps'):
					cli_name = '{}:{}'.format(address[0], address[1])
					if cli_name in g_client_fps:
						update_cli_fps(cli_name, t1)
					else:
						create_cli_fps(token_list[0], cli_name, t1)

		for cli_name,cli_fps in g_client_fps.iteritems():
			if ( (t1 - cli_fps['life_t']) > g_cli_fps_dead_dt * 1000.0):
				del g_datas[ cli_fps['d']['name'] ]
				del g_client_fps[cli_name]
				break

def process_input_gallery():

	g_gallery_state['browsing'] = False
	if (g_gallery_state['base_path'] is None):
		return

	def check_is_dir(path):
		try:
			if os.path.isdir(path):
				x = os.listdir(path)
				return True
		except:
			path
		return False

	def check_is_file(path):
		try:
			if os.path.isfile(path) and os.path.split(path)[1].endswith('.stxt'):
				return True
				#with open(path, 'r') as fi:
				#	return True if fi.readline().startswith('#') else False
		except:
			path
		return False

	def read_dirs(bpath):
		dirs = []; files = [];
		for x in os.listdir(bpath):
			xpath = os.path.join(bpath, x)
			if check_is_dir(xpath):
				dirs.append(x)
			elif check_is_file(xpath):
				files.append(x)

		parsed_one = False
		checked_files = []
		for file in files:
			xpath = os.path.join(bpath, file)
			if (check_is_file(xpath)):
				if (parsed_one) or (open_file_gallery(xpath) is not None):
					parsed_one = True
					checked_files.append(file)

		return (dirs, checked_files)

	def reset_gallery_path(path, dirs, files):
		g_gallery_state['curr_path'] = path
		g_gallery_state['curr_dirs_base'] = path
		g_gallery_state['curr_dirs'] = dirs; g_gallery_state['curr_dir_sel'] = 0;
		g_gallery_state['curr_files'] = files; g_gallery_state['curr_file_sel'] = 0;

	def load_gallery_files(first_file = 0, dir = 1):
		path = g_gallery_state.get('curr_path', None)
		files = g_gallery_state.get('curr_files', None)
		if (path and files):
			max_files = g_gallery_state['max_files']
			added_files = 0; fi = first_file; min_loaded = len(files); max_loaded = -1;
			while (fi >= 0 and fi < len(files)):
				if (graph_filter_allow(files[fi])):
					fdatas = open_file_gallery(os.path.join(path, files[fi]))
					if (fdatas):
						min_loaded = m_min(min_loaded, fi); max_loaded = m_max(max_loaded, fi);
						if (added_files == 0):
							g_gallery_state['loaded_files'] = []
							close_files_gallery()
						g_gallery_state['loaded_files'].append(fi)
						added_files = added_files + 1
						for d in fdatas:
							g_datas[d['name']] = d
						if (added_files >= max_files):
							break
				fi = fi + dir
			g_gallery_state['curr_file_first'] = min_loaded; g_gallery_state['curr_file_last'] = max_loaded;

	def down_dirs():
		if ('curr_dirs_base' not in g_gallery_state) or (g_gallery_state['curr_dirs_base'] != g_gallery_state['curr_path']):
			(dirs, files) = read_dirs(g_gallery_state['curr_path'])
			if (len(files) == 0 and len(dirs) == 1):
				g_gallery_state['curr_path'] = os.path.join(g_gallery_state['curr_path'], dirs[0])
				down_dirs()
			else:
				reset_gallery_path(g_gallery_state['curr_path'], dirs, files)

	def up_dirs_impl(new_path):
		if (len(os.path.join(new_path, 'a')) >= len(os.path.join(g_gallery_state['base_path'], 'a'))):
			(dirs, files) = read_dirs(new_path)
			if (len(files) == 0 and len(dirs) == 1):
				if (up_dirs_impl(os.path.dirname(new_path)) == False):
					reset_gallery_path(new_path, dirs, files)
			else:
				reset_gallery_path(new_path, dirs, files)
			return True
		else:
			return False

	def up_dirs():
		up_dirs_impl(os.path.dirname(g_gallery_state['curr_path']))

	def refresh_dirs():
		new_path = g_gallery_state['curr_path']
		if (new_path is not None) and (g_gallery_state.get('curr_dirs_base', None) != new_path):
			up_dirs_impl(new_path)

	if (g_gallery_state.get('browse_dirs', False)):
		for key in g_special_keys.keys():
			if (key == '101'):		#up
				g_gallery_state['curr_dir_sel'] = m_max(0, (g_gallery_state.get('curr_dir_sel',0)-1))
			elif (key == '103'):	#down
				g_gallery_state['curr_dir_sel'] = m_min(len(g_gallery_state['curr_dirs'])-1, g_gallery_state.get('curr_dir_sel',0)+1)
			elif (key == '100'):	#left
				up_dirs()
			elif (key == '102'):	#right
				if (len(g_gallery_state['curr_dirs']) > 0):
					new_path = os.path.join(g_gallery_state['curr_path'], g_gallery_state['curr_dirs'][g_gallery_state['curr_dir_sel']])
					g_gallery_state['curr_path'] = new_path;
					down_dirs()
	else:
		if (len(g_gallery_state.get('curr_files', [])) > 0):
			for key in g_special_keys.keys():
				if (key == '100'):		#left
					file_sel = m_max(g_gallery_state.get('curr_file_first', 0)-1, g_gallery_state['max_files']-1)
					load_gallery_files(file_sel, -1)
				elif (key == '102'):	#right
					file_sel = m_min(g_gallery_state.get('curr_file_last', -1)+1, m_max(0, len(g_gallery_state['curr_files'])-g_gallery_state['max_files']))
					load_gallery_files(file_sel, 1)

	if (g_frame == 0):
		refresh_dirs()
		if (len(g_gallery_state['curr_files']) == 0):
			g_gallery_state['browse_dirs'] = True
		else:
			load_gallery_files()

	if ('\r' in g_keys.keys()):
		g_gallery_state['browse_dirs'] = not g_gallery_state.get('browse_dirs', False)
		if (g_gallery_state['browse_dirs']):
			refresh_dirs()
		else:
			load_gallery_files()

	if (g_gallery_state.get('browse_dirs', False)):
		def cut_display_strings(items, sel):
			max_disp = 24
			disp_first_max = m_max(0, len(items) - max_disp)
			disp_first = m_min(m_max(0, sel), disp_first_max)
			cut_items = items[disp_first: m_min(disp_first+max_disp, len(items)) ]
			disp_cut_below = (disp_first > 0)
			disp_cut_above = (len(cut_items)+disp_first < len(items))
			if (sel >= 0):
				disp_strings = [' {}. {}[{}]'.format(i+disp_first+1, '>' if i == (sel-disp_first) else '',cut_items[i]) for i in range(len(cut_items))]
			else:
				disp_strings = [' [{}]'.format(cut_items[i]) for i in range(len(cut_items))]
			disp_strings = (['...'] if disp_cut_below else []) + disp_strings + (['...'] if disp_cut_above else [])
			return disp_strings

		g_gallery_state['browsing'] = True
		refresh_dirs()
		dirs = g_gallery_state['curr_dirs']; files = g_gallery_state['curr_files'];
		len_dirs = len(dirs); len_files = len(files);
		head_deco = ['[{}] {}'.format(g_gallery_state['curr_path'], '({})'.format(len_files) if len_files else '')]
		body_deco = []
		if (len_dirs == 0 and len_files > 0):
			body_deco = cut_display_strings(files, -1)
		else:
			body_deco = cut_display_strings(dirs, g_gallery_state.get('curr_dir_sel', 0))
		full_deco = head_deco + body_deco
		center = screen_to_draw([g_wind_ext[0]/2.0, g_wind_ext[1]/2.0], g_wind_ext[1])
		dlg_bound = defer_draw_strings(full_deco, center[0], center[1], [1.0]*3, g_wind_ext[1], 'cc', [0.3]*3)
		#trace_bound(dlg_bound)

	if (len(g_gallery_state.get('loaded_files', [])) > 0):
		top_center = screen_to_draw([g_wind_ext[0]/2.0, 0.0], g_wind_ext[1])
		loaded_string = '[{}-{} of {} from [{}]]'.format(g_gallery_state['curr_file_first'], g_gallery_state['curr_file_last'], len(g_gallery_state['curr_files']), os.path.split(g_gallery_state['curr_path'])[1])
		defer_draw_strings([loaded_string], top_center[0], top_center[1], [1.0]*3, g_wind_ext[1], 'ct')



def process_input_realtime(bgraphs, bstats):
	global g_keys
	global g_special_keys
	global g_cmd
	global g_long_cmd
	global g_pause
	global g_track
	global g_mouse
	global g_buttons
	global g_analyse_conv
	global g_freq_mode
	global g_mouse_graph
	global g_track_unit_conv
	global g_filtering
	global style_color3f

	mouse_pt = [-2.0, -2.0]
	mouse_btns = {}

	do_switch_represent = False

	if ('-dbg_keys' in sys.argv):
		if (len(g_keys)):
			print g_keys.keys()
		if (len(g_special_keys)):
			print g_special_keys.keys()

	if (g_frame == 0 or '1' in g_special_keys.keys()):
		print 'Realtime keys:'
		print ' [t]rack,[h]ide,[u]nhide,[b]lock,[k]ill,[F]it,[L]ife,[a]lternate,[d]ump,[r]epresent,[p]ause,[n]ow'
		print ' [F1] help, [F2] analyse, [F3] freq'
		print ' Mouse: [R]:select, [M]:exclusive'
		print ''

	for key in g_special_keys.keys():
		if (key == '2'):
			g_analyse_conv = not g_analyse_conv
		elif (key == '3'):
			g_freq_mode = not g_freq_mode
		elif (key == '4'):
			g_mouse_graph = not g_mouse_graph
		elif (key == '5'):
			style_color3f = style1_color3f if (style_color3f == style0_color3f) else style0_color3f
		elif (key == '9'):
			fkeys = g_unit_conv_funcs.keys()
			index = fkeys.index(g_track_unit_conv)
			g_track_unit_conv = fkeys[(index+1)%len(fkeys)]

	cmd_mouse_graph(g_mouse_graph)

	if (not g_long_cmd):
		process_input_gallery()

	if (g_gallery_state.get('browsing', False) == False):

		if ('\t' in g_keys.keys()):
			next_tab()
		cmd_stripped = [x for x in g_keys.keys() if ((ord(x) >= 32 and ord(x) <= 127) or (x == '\r') or (x == '\x08'))]
		cmd_add = ''.join(cmd_stripped); g_cmd = g_cmd+cmd_add;
		while (len(g_cmd) and g_cmd[-1] == '\x08'):
			g_cmd = g_cmd[:-2 if len(g_cmd) > 1 else -1]
		if (g_long_cmd and '`' in g_cmd):
			g_cmd = ''
			g_long_cmd = False
		cmds = g_cmd.split('\r')
		if (g_cmd.endswith('\r') or (not g_long_cmd)):
			g_cmd = ''
		else:
			g_cmd = cmds.pop()
		for cmd_str in cmds:
			cmd_parts = [cmd_str] if g_long_cmd else [x for x in cmd_str]
			for cmd in cmd_parts:
				if (len(cmd)):
					g_long_cmd = False
					if (len(cmd) > 1):
						lcmds = cmd.split(';')
						for lcmd in lcmds:
							acmd = lcmd.split()
							try:
								if (acmd[0].startswith('*')):
									datas = match_datas(g_datas, aget(acmd, 1, '*'), '')
									max = cmd_mul_len(datas, float(acmd[0][1:]), 8)
									print 'max len:', max
								elif (acmd[0] in ['e', 'export']):
									datas = match_datas(g_datas, acmd[1], '')
									cmd_export_datas(datas, aget(acmd, 2, ''), g_defaults.get('path', ''))
								elif (acmd[0] == 'path'):
									g_defaults['path'] = aget(acmd, 1, '')
								elif (acmd[0] == 'freq'):
									g_freq_mode = not g_freq_mode
								elif (acmd[0] == 'hz'):
									cmd_set_default_hz(float(aget(acmd, 1, 0.0)))
								elif (acmd[0] == 'dbg_conv'):
									g_defaults['dbg_conv'] = not g_defaults['dbg_conv']
								elif (acmd[0] == 'def'):
									print g_defaults
								elif (acmd[0] == 'conv'):
									g_track_unit_conv = aget(acmd, 1, 'id')
								elif (acmd[0] == 'only'):
									g_graph_filtering['only'] = [x for x in aget(acmd, 1, '').split(',') if len(x)]
								elif (acmd[0] == 'hide'):
									g_graph_filtering['hide'] = [x for x in aget(acmd, 1, '').split(',') if len(x)]
								elif (acmd[0] == 'all'):
									g_graph_filtering['only'] = []; g_graph_filtering['hide'] = [];
							except Exception as e:
								print traceback.format_exc() if g_dbg else str(e)
					else:
						if ('`' == cmd):
							g_long_cmd = not g_long_cmd
						elif ('p' == cmd):
							g_pause	= not g_pause
						elif ('n' == cmd):
							print 'Frame:{}, Fps:{:.0f}'.format(g_frame, g_fps)
						elif ('t' == cmd):
							g_track	= not g_track
						elif ('a' == cmd):
							for bg in bgraphs:
								if not (bg['data']['blocked'] or bg['hidden']):
									bg['select'] = not bg['select']
						elif ('d' == cmd):
							if (len(bstats)):
								print '\nStats:'
								a_stats = sorted(bstats, lambda x,y: x['data']['index']-y['data']['index'])
								for bs in a_stats:
									if (not(bs['data']['blocked'])):
										print get_stat_info(bs['rep'], bs['data'])

							if (len(bgraphs)):
								show_all = ('g' in cmd)
								a_graphs = sorted(bgraphs, lambda x,y: x['data']['index']-y['data']['index'])
								print '\nGraphs:'
								for bg in a_graphs:
									if (show_all or ((bg['select'] or bg['focus']) and not(bg['data']['blocked'] or bg['hidden'])) ):
										print get_graph_info(bg['rep'], bg['data'], bg)
						elif ('d' == cmd):
							for bg in bgraphs:
								bg['select'] = False
						elif ('u' == cmd):
							for bg in bgraphs:
								bg['hidden'] = False
						elif ('h' == cmd):
							for bg in bgraphs:
								if (bg['select'] or bg['focus']):
									bg['hidden'] = True
						elif ('b' == cmd):
							for bg in bgraphs:
								if (bg['select'] or bg['focus']):
									bg['data']['blocked'] = True
									bg['data']['pts'] = []
									bg['data']['miny'] = None
						elif ('k' == cmd):
							for bg in bgraphs:
								if (bg['select'] or bg['focus']):
									del g_datas[bg['data']['name']]
								(bgraphs, bstats) = get_bound_reps(g_datas, g_tab)
						elif ('L' == cmd or 'l' == cmd):
							life = 'L' in cmd
							cands = [x for x in bgraphs if (x['select'] or x['focus'])]
							if (len(cands) == 1):
								cands[0]['fit_life'] = not cands[0]['fit_life']
							else:
								for bg in cands:
									bg['fit_life'] = life
						elif ('F' == cmd or 'f' == cmd):
							fit = 'F' in cmd
							cands = [x for x in bgraphs if (x['select'] or x['focus'])]
							if (len(cands) == 1):
								cands[0]['fit'] = not cands[0]['fit']
							else:
								for bg in cands:
									bg['fit'] = fit
						elif ('r' == cmd):
							do_switch_represent = True
						elif ('m' == cmd):
							g_fast_mode = not g_fast_mode
						#else:
						#	if (g_long_cmd == False):
						#		g_long_cmd = True
						#		g_cmd = cmd

	g_keys = {}
	g_special_keys = {}
	state_select = True

	if g_long_cmd:
		center = screen_to_draw([g_wind_ext[0]/2.0, g_wind_ext[1]/2.0], g_wind_ext[1])
		defer_draw_strings(['cmd: {}'.format(g_cmd)], center[0], center[1], [1.0]*3, g_wind_ext[1], 'cc', [0.3]*3)

	if (g_mouse[0] is not None):
		 mouse_pt = screen_to_draw(g_mouse, g_wind_ext[1])
	for gb in g_buttons.values():
		if (len(gb)):
		 	mouse_btns[gb['button']] = gb; mouse_btns[gb['button']]['pt'] =  screen_to_draw(gb['wpt'], g_wind_ext[1]);
	g_buttons = {}

	return (state_select, do_switch_represent, mouse_pt, mouse_btns)

def display(w, h):
	global g_pix_to_view_scale
	global g_mouse
	global g_buttons
	global g_keys
	global g_special_keys
	global g_track
	global g_state
	global g_exclusive
	global g_pause
	global g_frame
	global g_fast_mode
	global g_fps
	global g_fps_t0
	global g_fps_frames
	global g_cmd
	global g_long_cmd
	global g_defaults
	global g_freq_mode
	global g_analyse_conv
	global g_wind_ext
	global g_mouse_graph

	g_wind_ext = [w, h]
	t0 = glutGet(GLUT_ELAPSED_TIME)
	g_frame = g_frame+1

	process_sock_realtime()
	process_sock_export(False)

	if ('\x1b' in g_keys.keys()):
		#glutLeaveMainLoop()
		sys.exit(0)

	if (g_pause):
		if ('p' not in g_keys.keys() and ' ' not in g_keys.keys()):
			return
		else:
			if (' ' in  g_keys.keys()):
				del g_keys[' ']

	for d in g_datas.values():
		process_data_pts(d, g_reps)

	g_pix_to_view_scale = [1.0/w, 1.0/h]
	aspect = float(w)/float(h)
	glViewport(0, 0, w, h)
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	glOrtho(-1 * aspect, 1 * aspect, -1, 1, -1, 1)

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity()

	(r,g,b) = style_color3f(0,0,0)
	glClearColor(r,g,b, 0.0)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

	(bgraphs, bstats) = get_bound_reps(g_datas, g_tab)

	#time.sleep(0.1)
	(state_select, do_switch_represent, mouse_pt, mouse_btns) = process_input_realtime(bgraphs, bstats)

	if 1:
		s_graphs = bgraphs
		if 1:
			if (g_exclusive is not None) and (g_exclusive in g_datas) and (g_datas[g_exclusive]['bind']) and (g_datas[g_exclusive]['bind']['rep']['rtype'] == 'g'):
				eg = g_datas[g_exclusive]['bind']['rep']
				if (eg['window'] is None):
					s_graphs = [g_datas[g_exclusive]['bind']]
				else:
					wnd = eg['window']
					s_graphs = [g for g in s_graphs if g['rep']['window'] == wnd]
		s_graphs = sorted(s_graphs, lambda x,y: x['data']['index']-y['data']['index'])
		s_graphs = [bg for bg in s_graphs if (bg['hidden'] == False and bg['data']['blocked'] == False and graph_filter_allow(bg['data']['name']) )]
		a_graphs = []
		processed = {}
		for bgi in range(len(s_graphs)):
			if (bgi not in processed):
				w_graphs = [bgi]
				if (s_graphs[bgi]['rep']['window'] is not None):
					wind = s_graphs[bgi]['rep']['window']
					for bgj in range(bgi+1, len(s_graphs)):
						if (s_graphs[bgj]['rep']['window'] == wind):
							w_graphs.append(bgj)
				for bwgi in w_graphs:
					processed[bwgi] = True
				a_graphs.append([s_graphs[bwgi] for bwgi in w_graphs])

	upper_off =	1.1 * get_string_height(); upper_off = 0 if h <= upper_off else upper_off;
	wbounds = create_bounds(len(a_graphs), w, h-upper_off, 0, upper_off)
	bounds = []
	for wb in wbounds:
		bound = make_bound(screen_to_draw(wb.min, h), screen_to_draw(wb.max, h))
		bounds.append(bound)

	margin = [ 2.0 / w, 2.0 / h ]
	margin = v2_muls(margin, 3)
	margin_bounds = create_margins(bounds, margin)

	select_btn = 2
	exclusive_btn = 1
	for bag,b in zip(a_graphs, margin_bounds):
		focus = is_in_bound(mouse_pt, b)
		switch_exclusive = (exclusive_btn in mouse_btns and mouse_btns[exclusive_btn]['mode'] == 1 and is_in_bound(mouse_btns[exclusive_btn]['pt'], b))
		has_exclusive = False
		for bg in bag:
			col = [0.8, 0.3, 0.0] if (bg['select']) else [1.0, 1.0, 1.0]
			if (focus):
				col = [0.9,0.4,0.0] if (bg['select']) else [1.0,0.5,0.0]
				#if (not bg['focus']):
				#	print get_graph_info(bg['rep'], bg['data'], bg)
			if (state_select):
				if (select_btn in mouse_btns and mouse_btns[select_btn]['mode'] == 1 and is_in_bound(mouse_btns[select_btn]['pt'], b)):
					bg['select'] = not bg['select']
			bg['focus'] = focus
			has_exclusive = has_exclusive or bg['data']['name'] == g_exclusive
		if (switch_exclusive):
			if (has_exclusive):
				g_exclusive = None
			else:
				g_exclusive = bag[0]['data']['name']
		trace_bound(b, col)


	gcols = [xcol(255,127,127), xcol(127,255,127), xcol(127,127,255)]
	for bag,b in zip(a_graphs, margin_bounds):
		for bgi in range(len(bag)):
			bg = bag[bgi]
			if (len(bag) > 1):
				col = gcols[bgi%len(gcols)]
			else:
				col = [1.0, 1.0, 1.0] if 'col' not in bg['rep'] else bg['rep']['col']
			draw_graph(bg['rep'], bg['data'], bg, b, bg['focus'] and g_track and g_mouseFocus, mouse_pt, col, g_fast_mode, g_freq_mode, g_analyse_conv or bg['focus'])

	selected_bstat = None
	if (len(bstats)):
		s_stats = sorted(bstats, lambda x,y: x['data']['index']-y['data']['index'])
		s_stats = [x for x in s_stats if (not x['data']['blocked'])]
		left_top = screen_to_draw([5.0, 3.0], h)
		strings = [make_stat_string(x['rep'], x['data']) for x in s_stats]
		sbounds = draw_strings(strings, left_top[0], left_top[1], [1.0]*3, h)
		for bi in range(len(sbounds)):
			b = sbounds[bi]
			if (is_in_bound(mouse_pt, b)):
				selected_bstat = s_stats[bi]
				trace_bound(b, [0.8, 0.3, 0.0])

	if (len(g_tabs) and len(g_tab)):
		center_top = screen_to_draw([w/2, 3.0], h)
		draw_strings(['[{}]'.format(g_tab)], center_top[0], center_top[1], [1.0]*3, h)

	if 1:
		if (do_switch_represent):
			if (selected_bstat is not None):
				d = selected_bstat['data']
				if (d['is_ordered']):
					r = None
					if (d['last_bind'] and d['last_bind']['rep']['rtype'] == 'g'):
						r = d['last_bind']['rep']
					else:
						r = g_reps['_default_g']
					if (r['name'] == '_default_g'):
						set_data_proc(d, g_defaults['g_proc'])
					bind_data(d, r)
			else:
				for bg in bgraphs:
					if (bg['select'] or bg['focus']):
						d = bg['data']
						r = None
						if (d['last_bind'] and d['last_bind']['rep']['rtype'] == 's'):
							r = d['last_bind']['rep']
						else:
							r = g_reps['_default_s']
						if (r['name'] == '_default_s'):
							set_data_proc(d, g_defaults['s_proc'])
						bind_data(d, r)


	flush_defer_draw_strings()

	glutSwapBuffers()
	t1 = glutGet(GLUT_ELAPSED_TIME)
	g_fps_frames = g_fps_frames+1
	if (g_fps_t0 <= 0) or (t1-g_fps_t0 >= 1000.0):
		g_fps = (1000.0 * float(g_fps_frames)) / float(t1-g_fps_t0)
		g_fps_t0 = t1; g_fps_frames = 0;
	#print (t1-t0)

def reshape(w, h):
	glutDisplayFunc(lambda: display(w, h))
	glutPostRedisplay();

def sys_argv_has(keys):
	if (hasattr(sys, 'argv')):
		for i in range(len(keys)):
			 if (keys[i] in sys.argv):
				return True
	return False

def sys_argv_has_key(keys):
	if ( hasattr(sys, 'argv')):
		for key in keys:
			ki = sys.argv.index(key) if key in sys.argv else -1
			if (ki >= 0 and ki+1 < len(sys.argv)):
				return True
	return False

def sys_argv_get(keys, dflt):
	if ( hasattr(sys, 'argv')):
		for key in keys:
			ki = sys.argv.index(key) if key in sys.argv else -1
			if (ki >= 0 and ki+1 < len(sys.argv)):
				return sys.argv[ki+1]
	return dflt

def add_tests():
	global g_reps

	def test_update1(d):
		x = float(d['in'])
		add_data_pt(d, math.sin(0.05*x) / (0.1*(x+1)))

	r = set_graph(g_reps, '_test_g1', 'y', 1, -2.0, 2.0, True); d = set_data(g_datas, '_test_g1', 'f', 'n512', 'g'); d['update'] = test_update1; bind_data(d, r);

	def test_update1a_1hz(d):
		x = float(d['in']); add_data_pt(d, math.sin(2*3.14*x/d['len']));

	def test_update1a(d):
		x = float(d['in']); add_data_pt(d, math.sin(0.08*x));

	if 1:
		add_tab('test_tab')
		for i in range(5):
			r = set_graph(g_reps, '_test_g1_{}'.format(i), 'y', 1, -2.0, 2.0, i%2); d = set_data(g_datas, '_test_g1_{}'.format(i), 'f', 'n128', 'g'); d['update'] = test_update1a if i%2 else test_update1a_1hz; bind_data(d, r);
			d['tab'] = 'test_tab'

	def test_update1b(d):
		x = float(d['in']); add_data_pt(d, 1.0/(x+1));

	r = set_graph(g_reps, '_test_g1b', 'y', 1, -2.0, 2.0, True); d = set_data(g_datas, '_test_g1b', 'f', 'n512', 'g'); d['update'] = test_update1b; bind_data(d, r);
	d['tab'] = 'test_tab'

	def test_update1c(d):
		x = float(d['in']); add_data_pt(d, x*x);

	r = set_graph(g_reps, '_test_g1c', 'y', 1, -2.0, 2.0, True); d = set_data(g_datas, '_test_g1c', 'f', 'n512', 'g'); d['update'] = test_update1c; bind_data(d, r);
	d['tab'] = 'test_tab'

	if 1:
		r = set_graph(g_reps, '_test_g2', 'y', 0, -1.5, 0.2, True); d = set_data(g_datas, '_test_g2', 'f', 'n512', 'g'); d['update'] = test_update1; bind_data(d, r);
		r = set_graph(g_reps, '_test_g3', 'y', 1, -1.0, 1.0, False); d = set_data(g_datas, '_test_g3', 'f', 'n512', 'g'); d['update'] = test_update1; bind_data(d, r);
		g_reps['_test_g2']['window'] = 'share1'; g_reps['_test_g3']['window'] = 'share1';
		for i in range(2):
			n = '_test_g3_{}'.format(i)
			r = set_graph(g_reps, n, 'y', 1, 0.0, 1.0, True); d = set_data(g_datas, n, 'f', 'n512', 'g'); d['update'] = test_update1; bind_data(d, r);

		if 1:
			def test_update4(d,sx,sy):
				x = float(d['in'])
				sclx = math.log(1+x)/sx; scly = math.log(1+x)/sy;
				add_data_pt(d, [sclx * math.sin(0.05*x),scly * math.cos(0.05*x)] )

			r = set_graph(g_reps, '_test_xy0', 'xy', 0, -1.5, 1.5, True); d = set_data(g_datas, '_test_xy0', 'vf', 'n512', 'g'); d['update'] = lambda g:test_update4(g, 5.0, 5.0); bind_data(d, r);
			r = set_graph(g_reps, '_test_xy1', 'xy', 0, -1.5, 1.5, True); d = set_data(g_datas, '_test_xy1', 'vf', 'n512', 'g'); d['update'] = lambda g:test_update4(g, 7.0, 7.0); bind_data(d, r);
			r = set_graph(g_reps, '_test_xy2', 'xy', 0, -1.5, 1.5, True); d = set_data(g_datas, '_test_xy2', 'vf', 'n512', 'g'); d['update'] = lambda g:test_update4(g, 9.0, 5.0); bind_data(d, r);
			g_reps['_test_xy0']['window'] = 'share2'; g_reps['_test_xy1']['window'] = 'share2'; g_reps['_test_xy2']['window'] = 'share2';

			def test_update5a(d):
				x = float(d['in'])
				add_data_pt(d, math.sin(0.25*x))

			r = set_stat(g_reps, '_test2_x1', 'n1'); d = set_data(g_datas, '_test2_x1', 'f', 'n64', 's'); d['update'] = test_update5a; bind_data(d, r);

			def test_update5(d):
				add_data_pt(d, d['in'])
			r = set_stat(g_reps, '_test_x1', 'n1'); d = set_data(g_datas, '_test_x1', 'i', 'n64', 's'); d['update'] = test_update5; bind_data(d, r);

	if 1:
		if 1:
			r = set_stat(g_reps, '_test_x4', 'n4'); d = set_data(g_datas, '_test_x4', 'i', 'n64', 's'); d['update'] = test_update5; bind_data(d, r);
			r = set_stat(g_reps, '_test_l3', 'p3'); d = set_data(g_datas, '_test_l3', 'i', 'n64', 's'); d['update'] = test_update5; bind_data(d, r);

		def test_update6(d):
			add_data_pt(d, str(float(d['in'])))
		r = set_stat(g_reps, '_test_ls1', 'p1'); d = set_data(g_datas, '_test_ls1', 's', 's', 's'); d['update'] = test_update6; bind_data(d, r);

		if 1:
			def test_update7(d):
				if (test_update7 not in d):
					d[test_update7] = 0
				if (d[test_update7] % 32 == 0):
					add_data_pt(d, 1)
				d[test_update7] = d[test_update7]+1
			r = set_stat(g_reps, '_test_+'); d = set_data(g_datas,'_test_', 'i', '+', 's'); d['update'] = test_update7; bind_data(d, r);


			def test_update8(d,sx,sy):
				x = float(d['in'])
				sclx = math.log(1+x)/sx; scly = math.log(1+x)/sy;
				add_data_pt(d, [sclx * math.sin(0.05*x),scly * math.cos(0.05*x)] )
			r = set_stat(g_reps, '_test_sl', 'n1'); d = set_data(g_datas,'_test_sl', 'vf', 's', 's'); d['update'] = lambda s:test_update8(s, 9.0, 5.0); bind_data(d, r);

def bind_socket(ip, port):
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind((ip, port))
	sock.setblocking(0)
	return sock

def main_realtime():
	global g_reps
	global g_sock
	global d_filters
	global d_defaults

	test = False
	do_profile = False
	test = sys_argv_has(['-test'])
	g_defaults['dbg_conv'] = sys_argv_has(['-dbg_conv'])
	do_profile = sys_argv_has(['-profile'])
	d_filters = [x for x in sys_argv_get(['-filter'], '').split('/') if len(x)]

	g_defaults['s_proc'] = 'n{}'.format(int(sys_argv_get(['-s_len'], g_defaults['s_proc'][1:])))
	g_defaults['g_proc'] = 'n{}'.format(int(sys_argv_get(['-g_len', '-n', '-len'], g_defaults['g_proc'][1:])))
	set_graph(g_reps, '_default_g', 'y', True, -100.0, 100.0, True)
	set_stat(g_reps, '_default_s')

	ip = sys.argv[sys.argv.index('-ip')+1] if '-ip' in sys.argv else "127.0.0.1"
	port = int(sys.argv[sys.argv.index('-port')+1] if '-port' in sys.argv else "17641")

	if test:
		add_tests()
	else:
		g_sock = bind_socket(ip, port)
		#prepare_export(17642)

	if (prepare_gallery() == False):
		return 0

	glutInit(sys.argv)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
	glutInitWindowSize(800, 600)
	glutCreateWindow('trace')

	if do_profile:
		for i in range(100):
			display(800, 600)
		import cProfile, pstats, StringIO
		pr = cProfile.Profile()
		pr.enable()
		for i in range(100):
			display(800, 600)
		pr.disable()
		s = StringIO.StringIO()
		ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
		ps.print_stats(24)
		print s.getvalue()
	else:
		glutReshapeFunc(reshape)
		glutIdleFunc(glutPostRedisplay)
		glutMouseFunc(handleMouseAct)
		glutPassiveMotionFunc(handleMousePassiveMove)
		glutMotionFunc(handleMouseMove)
		glutEntryFunc(handleMouseEntry)
		glutKeyboardFunc(handleKeys)
		glutSpecialFunc(handleSpecialKeys)
		glutMainLoop()

def get_realtime_usage():
	return 'realtime mode: [-test] [-dbg] [-h]'

def main():
	global g_dbg
	g_dbg = sys_argv_has(['-dbg'])

	if (sys_argv_has(['-h', '-help', '--h', '--help'])):
		print get_realtime_usage()
		print get_gallery_usage()
		print get_export_usage()
		print ''

	if (sys_argv_has(['-export'])):
		main_export()
	else:
		main_realtime()

main()
