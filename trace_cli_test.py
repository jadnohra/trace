import sys, socket, struct, math, time

def main():
	g_graphs = {}
	g_dbg = '-dbg' in sys.argv
	ip = sys.argv[sys.argv.index('-ip')+1] if '-ip' in sys.argv else "127.0.0.1"
	cli_ip = sys.argv[sys.argv.index('-cli_ip')+1] if '-cli_ip' in sys.argv else ""
	port = int(sys.argv[sys.argv.index('-port')+1] if '-port' in sys.argv else "17641")
	it = int(sys.argv[sys.argv.index('-it')+1] if '-it' in sys.argv else "5")
	ms = int(sys.argv[sys.argv.index('-ms')+1] if '-ms' in sys.argv else "0")

	g_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	g_sock.setblocking(0)

	if len(cli_ip):
		g_sock.bind((cli_ip, 0))

	def sock_send(msg):
		g_sock.sendto(msg, (ip, port))
		if g_dbg:
			print msg
	def set_graph(graphs, name, type, xpts, fit, miny, maxy, lines):
		graphs[name] = {}; graphs[name]['name'] = name;
		#sock_send("graph {} {} {} {} {} {} {}".format(name, type, xpts, int(fit), f2h(miny), f2h(maxy), int(lines)))
	def add_graph_pt(graphs, name, pt):
		sock_send("pt {} {} {}".format(name, 'f', str(pt)))
	def test_update1(g):
		x = float(g['in'])
		add_graph_pt(g_graphs, g['name'], math.sin(0.05*x) / (0.1*(x+1)))

	set_graph(g_graphs, '_test_g1', 'y', 256, 1, -2.0, 2.0, True); g_graphs['_test_g1']['update'] = test_update1;

	if ('-echo' in sys.argv):
		sock_send('echo hello')
		time.sleep(ms/1000.0)
		try:
			data,address = g_sock.recvfrom(128)
			print 'Received as echo: [{}]'.format(data)
		except Exception:
			print 'no answer'

	if ('-it' not in sys.argv):
		return

	for x in range(it):
		for gg in g_graphs.values():
			gg['in'] = gg.get('in', 0) + 1
			upd = gg.get('update')
			if (upd):
				gg['update'](gg)
		if ms > 0:
			time.sleep(ms/1000.0)

main()
