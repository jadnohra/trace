#ifndef jad_trace_h
#define jad_trace_h


#ifdef _WIN32
	#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
		#define TRACE_WIN32
		#define TRACE_BSD
	#else
		#define TRACE_WINRT
	#endif
#else
	#define TRACE_BSD
#endif

#ifdef _WIN32
	#ifdef TRACE_WIN32
		#include <windows.h>
		#include <winsock.h>
		#pragma comment(lib,"wsock32.lib")
	#endif
	#ifdef TRACE_WINRT
		#include <ppl.h>
		#include <ppltasks.h>
		using namespace Windows::Networking;
		using namespace Windows::Networking::Sockets;
		using namespace Windows::Storage::Streams;
	#endif
#else
	// Normal BSD socket:
	#	include <sys/types.h>
	#	include <sys/time.h>
	#	include <sys/socket.h>
	#   include <sys/ioctl.h>
	#	include <netinet/in.h>
	#	include <netinet/tcp.h>
	#	include <arpa/inet.h>
	#	include <unistd.h>
	#	include <netdb.h>
	#	include <string.h>
	#	define closesocket close
	#	define INVALID_SOCKET (-1)
	#	define SOCKET_ERROR (-1)
#endif
#include "stdio.h"

namespace trace
{

enum ServerType { HexFloat, Float, Int, Str, HexDouble, HexFloat3, HexDouble3, HexFloat2, HexDouble2 };

class Client
{
public:

	inline static const char* def_ip() { return "127.0.0.1"; }
	inline static int def_port() { return 17641; }


	#ifdef TRACE_WINRT
		static std::wstring stows(std::string s)
		{
			std::wstring ws;
			ws.assign(s.begin(), s.end());
			return ws;
		}
	#endif

	Client()
	{
		 m_conCount = 0;
#ifdef TRACE_BSD
		m_socket = INVALID_SOCKET;
#endif
#ifdef TRACE_WINRT
		has_writer = false;
#endif
	}

	inline void disconnect()
	{
#ifdef TRACE_BSD
		if (m_socket != INVALID_SOCKET)
		{
			closesocket(m_socket);
			m_socket = INVALID_SOCKET;
			#ifdef WIN32
			WSACleanup();
			#endif
		}
#endif
#ifdef TRACE_WINRT
		has_writer = false;
		socket = nullptr;
		writer = nullptr;
#endif
	}

	inline int concount()
	{
		return m_conCount;
	}

	inline void connect(const char* host = def_ip(), const int port = def_port())
	{
		m_conCount++;

		disconnect();

#ifdef TRACE_BSD
		#ifdef WIN32
		WORD wVersionRequested = MAKEWORD(2, 2); WSADATA wsaData; int err;
		err = WSAStartup(wVersionRequested, &wsaData);
		#endif

		m_socket = socket(AF_INET, SOCK_DGRAM, 0);

		if (m_socket == INVALID_SOCKET)
		{
			#ifdef WIN32
			WSACleanup();
			#endif
			printf("tracer not connected\n");
			return;
		}

 		memset(&m_server, 0, sizeof(m_server));
		m_server.sin_family = AF_INET;
		m_server.sin_addr.s_addr = inet_addr(host);
		m_server.sin_port = htons(port);

		printf("tracer connected\n");
#endif
#ifdef TRACE_WINRT
		socket = ref new DatagramSocket();
		char sport[32]; sprintf(sport, "%d", port);

		HostName^ hostname = ref new HostName(ref new Platform::String(stows(host).c_str()));
		Platform::String^ serviceName = ref new Platform::String(stows(sport).c_str());
		get_writer_async = true;
		Concurrency::create_task(socket->ConnectAsync(hostname, serviceName)).then(
			[this](Concurrency::task<void> previousTask)
		{
			try
			{
				previousTask.get();
				writer = ref new DataWriter(socket->OutputStream);
				has_writer = true;
				get_writer_async = false;
				OutputDebugStringA("tracer connected\n");
			}
			catch (...)
			{
				get_writer_async = false;
				OutputDebugStringA("tracer not connected\n");
			}
		});

		while(get_writer_async)
		{
			Sleep(0);
		}
#endif
	}

	#ifdef TRACE_BSD
		inline bool has_socket() { return m_socket != INVALID_SOCKET; }
		int sock_send(SOCKET s, const char* buf, int len, int flags, const sockaddr* to, int tolen)
		{
			return sendto(s, buf, len, flags, to, tolen);
		}
		int sock_send(const char* buf, int len, int flags)
		{
			return sendto(m_socket, buf, len, flags, (struct sockaddr *) &m_server, sizeof(m_server));
		}
	#endif

	#ifdef TRACE_WINRT
		inline bool has_socket() { return has_writer; }
		int sock_send(const char* buf, int len, int flags)
		{
			Platform::Array<byte>^ bytes = ref new Platform::Array<byte>( (byte*) buf, len);
			try
			{
				writer->WriteBytes(bytes);
				writer->StoreAsync();
			}
			catch (...)
			{
				return 0;
			}
			return len;
		}
	#endif

	enum { BufSize = 128 };

	struct SockBuf
	{
		char buf[BufSize];
	};

	static unsigned int f2h(float f)
	{
		return *((unsigned int*) &f);
	}

	static unsigned long d2h(double d)
	{
		return *((unsigned long*) &d);
	}

	template<typename T> static inline const char* resolve_point(T y, ServerType stype, char* buf)
	{
		const char* type = 0;
		switch(stype)
		{
			case HexFloat: type = "hf"; sprintf(buf, "%08x", Client::f2h((float)y)); break;
			case HexDouble: type = "hd"; sprintf(buf, "%016x", Client::d2h((double)y)); break;
			case Float: type = "f"; sprintf(buf, "%f", (float)y); break;
			case Int: type = "i"; sprintf(buf, "%d", (int)y); break;
			//case Str: type = "s"; sprintf(buf, "%s", (const char*)y); break;
		}
		return type;
	}

	template<typename T> static inline const char* resolve_point2(T* y, ServerType stype, char* buf)
	{
		const char* type = 0;
		switch(stype)
		{
			case HexFloat2: type = "vhf"; sprintf(buf, "%08x,%08x", Client::f2h((float)y[0]), Client::f2h((float)y[1]) ); break;
			case HexDouble2: type = "vhd"; sprintf(buf, "%016x,%016x", Client::d2h((float)y[0]), Client::d2h((float)y[1]) ); break;
		}
		return type;
	}

	template<typename T> static inline const char* resolve_point3(T* y, ServerType stype, char* buf)
	{
		const char* type = 0;
		switch(stype)
		{
			case HexFloat3: type = "vhf"; sprintf(buf, "%08x,%08x,%08x", Client::f2h((float)y[0]), Client::f2h((float)y[1]), Client::f2h((float)y[2]) ); break;
			case HexDouble3: type = "vhd"; sprintf(buf, "%016x,%016x,%016x", Client::d2h((float)y[0]), Client::d2h((float)y[1]), Client::d2h((float)y[2]) ); break;
		}
		return type;
	}

	inline void server_echo(SockBuf& buf, const char* msg)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "echo %s", msg);
		server_send(buf.buf);
	}

	inline void server_print(SockBuf& buf, const char* msg)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "prnt %s", msg);
		server_send(buf.buf);
	}

	inline void server_gfps(SockBuf& buf)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "g_fps");
		server_send(buf.buf);
	}

	inline void server_sfps(SockBuf& buf)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "s_fps");
		server_send(buf.buf);
	}

	inline void server_point(SockBuf& buf, const char* ptcmd, const char* name, const char* type, const char* procstr, const char* y, const char* bind)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "%s %s %s %s %s %s", ptcmd, name, type, procstr, y, bind);
		server_send(buf.buf);
	}

	inline void server_graph(SockBuf& buf, const char* name, const char* type, const bool fit, const float miny, const float maxy, const bool lines)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "graph %s %s %d %08x %08x %d", name, type, int(fit), f2h(miny), f2h(maxy), int(lines));
		server_send(buf.buf);
	}

	inline void server_stat(SockBuf& buf, const char* name, const char* type)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "stat %s %s", name, type);
		server_send(buf.buf);
	}

	inline void server_bind(SockBuf& buf, const char* names)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "bind %s", names);
		server_send(buf.buf);
	}

	inline void server_unbind(SockBuf& buf)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "unbind");
		server_send(buf.buf);
	}

	inline void server_hz(SockBuf& buf, float freq)
	{
		if (!has_socket()) return;

		sprintf(buf.buf, "hz %f", freq);
		server_send(buf.buf);
	}

	inline void server_send(const char* buf)
	{
		int len =(int) strlen(buf);
		if (sock_send(buf, len, 0) != len)
		{
			printf("tracer fail\n");
		}
	}


	int m_conCount;
#ifdef TRACE_BSD
	size_t m_socket;
	sockaddr_in m_server;
#endif
#ifdef TRACE_WINRT
	DatagramSocket^ listener;
	DatagramSocket^ socket;
	DataWriter^ writer;
	bool has_writer;
	volatile bool get_writer_async;
#endif

	inline static Client& inst()
	{
		static Client kInst;
		return kInst;
	}

	inline static SockBuf& getBuf(int i=0)
	{
		static SockBuf buf[3];
		return buf[i];
	}
};

typedef Client::SockBuf SockBuf;

inline void connect(const char* ip = Client::def_ip(), int port = Client::def_port())
{
	Client::inst().connect(ip, port);
}

inline bool first_connect(const char* ip = Client::def_ip(), int port = Client::def_port())
{
	bool is_first = (Client::inst().concount() == 0);
	if (is_first)
	{
		connect(ip, port);
	}
	return is_first;
}

inline void disconnect()
{
	Client::inst().disconnect();
}

inline Client& cli()
{
	if (Client::inst().m_conCount == 0)
		connect();
	return Client::inst();
}

inline char* nameBuf(int i = 0) { static SockBuf buf[2]; return buf[i].buf; }

inline void echo(const char* msg)
{
	Client::inst().server_echo(Client::getBuf(), msg);
}

inline void print(const char* msg)
{
	Client::inst().server_print(Client::getBuf(), msg);
}

inline void sfps()
{
	Client::inst().server_sfps(Client::getBuf());
}

inline void gfps()
{
	Client::inst().server_gfps(Client::getBuf());
}

template<typename T> inline void point(const char* name, T y, ServerType stype, const char* procstr, const char* bind)
{
	const char* type = Client::resolve_point(y, stype, Client::getBuf(1).buf);
	if (type) cli().server_point(Client::getBuf(), "xpt", name, type, procstr, Client::getBuf(1).buf, bind);
}

template<typename T> inline void point(const char* name, T y, ServerType stype, const char* bind = "")
{
	const char* type = Client::resolve_point(y, stype, Client::getBuf(1).buf);
	if (type) cli().server_point(Client::getBuf(), "pt", name, type, "", Client::getBuf(1).buf, bind);
}

template<typename T> inline void point2(const char* name, T* y, ServerType stype, const char* bind = "")
{
	const char* type = Client::resolve_point2(y, stype, Client::getBuf(1).buf);
	if (type) cli().server_point(Client::getBuf(), "pt", name, type, "", Client::getBuf(1).buf, bind);
}


template<typename T> inline void point3(const char* name, T* y, ServerType stype, const char* bind = "")
{
	const char* type = Client::resolve_point3(y, stype, Client::getBuf(1).buf);
	if (type) cli().server_point(Client::getBuf(), "pt", name, type, "", Client::getBuf(1).buf, bind);
}

template<typename T> inline void spoint(const char* name, T y, ServerType stype, const char* bind = "")
{
	const char* type = Client::resolve_point(y, stype, Client::getBuf(1).buf);
	if (type) cli().server_point(Client::getBuf(), "s_pt", name, type, "", Client::getBuf(1).buf, bind);
}

template<typename T> inline void ppoint(const char* name, T y, ServerType stype, const char* bind = "")
{
	const char* type = Client::resolve_point(y, stype, Client::getBuf(1).buf);
	if (type) cli().server_point(Client::getBuf(), "+_pt", name, type, "", Client::getBuf(1).buf, bind);
}

inline void graph(const char* name, const char* type="y", const bool fit=true, const float miny=-100.0f, const float maxy=1000.0f, const bool lines = true)
{
	cli().server_graph(Client::getBuf(), name, type, fit, miny, maxy, lines);
}

inline void stat(const char* name, const char* type = "n1")
{
	cli().server_stat(Client::getBuf(), name, type);
}

inline void bind(const char* names)
{
	cli().server_bind(Client::getBuf(), names);
}

inline void unbind()
{
	cli().server_unbind(Client::getBuf());
}

inline void hz(float freq)
{
	cli().server_hz(Client::getBuf(), freq);
}

inline void test1()
{
	for (float i=0; i<128.0f; i+=1.0f)
	{
		trace::point("sin", sinf(i/32.0f), trace::HexFloat);
 		trace::point("cos", cosf(i/32.0f), trace::HexFloat);
		trace::point("sin+cos", sin(i/32.0f) + 0.2f*cosf(i/2.0f), trace::HexFloat);
	}
}

} // namespace trace

#ifdef WIN32
namespace trace {

typedef LONGLONG Time;

struct Clock
{
	double ifreq;
	double ifreq_milli;
	double ifreq_micro;
	double ifreq_nano;

	Clock()
	{
		LARGE_INTEGER f; QueryPerformanceFrequency(&f);
		ifreq = 1.0 / double(f.QuadPart);
		ifreq_milli = ifreq * 1.e3;
		ifreq_micro = ifreq * 1.e6;
		ifreq_nano = ifreq * 1.e9;
	}

	inline Time time()
	{
		LARGE_INTEGER i;
		QueryPerformanceCounter(&i);
		return i.QuadPart;
	}

	double toSecs(const Time& t) { return double(t) * ifreq; }
	double toMillis(const Time& t) { return double(t) * ifreq_milli; }
	double toMicros(const Time& t) { return double(t) * ifreq_micro; }
	double toNanos(const Time& t) { return double(t) * ifreq_nano; }
	static Clock& inst() { static Clock clock; return clock; }
};

static Clock& ClockInst() { return Clock::inst(); }
inline Time time() { return Clock::inst().time(); }
inline double toSecs(const Time& t) { return Clock::inst().toSecs(t); }
inline double toMillis(const Time& t) { return Clock::inst().toMillis(t); }
inline double toMicros(const Time& t) { return Clock::inst().toMicros(t); }
inline double toNanos(const Time& t) { return Clock::inst().toNanos(t); }

struct TimeInterval
{
	Time t;
	inline void start() { t = time(); }
	inline void stop() { t = time() - t; }
};

inline void test2()
{
	for (float i=0; i<128.0f; i+=1.0f)
	{
		trace::TimeInterval t;
		{ t.start(); Sleep(30); t.stop(); }
		trace::point("time", (float) toMillis(t.t), trace::Float);
	}
}

} // namespace trace
#endif WIN32

#endif // jad_trace_h
