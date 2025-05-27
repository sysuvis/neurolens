#pragma once
#include <cpprest/http_client.h>
#include <cpprest/json.h>  
#include <cpprest/filestream.h>
#include <iostream>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams

pplx::task<void> HTTPGetAsync()
{

	http_client client(U("http://127.0.0.1:5080"));
	uri_builder builder(U("/api/data"));
	// Append the query parameters: ?method=flickr.test.echo&name=value
	builder.append_query(U("q"), U("cpprestsdk github"));
	auto path_query_fragment = builder.to_string();
	// Make an HTTP GET request and asynchronously process the response
	return client.request(methods::GET, path_query_fragment).then([](http_response response)
		{
			// Display the status code that the server returned
			std::wostringstream stream;
			stream << L"Server returned returned status code " << response.status_code() << L'.' << std::endl;

		});
}

