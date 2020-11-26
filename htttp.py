# -*- coding: utf-8 -*-

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
from urllib.parse import urlparse
import json
import cgi
import ssl


data = {"isAuto":True,"isLightSensor":True,"lightValue":80,"lightSensorValue":1000}

class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        global data
        try:
            parsed_path = urlparse(self.path)
            send_data = json.dumps(data)
            if(parsed_path.path =='/data'):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(bytes(send_data, 'utf-8') )

        except:
            self.send_response(400)
            self.end_headers()

    def do_POST(self):
        # try:
            if self.path.endswith("/mode"):
                ctype, pdict = cgi.parse_header(self.headers['content-type'])
                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                if ctype == 'multipart/form-data':
                    fields = cgi.parse_multipart(self.rfile, pdict)
                    print("Fields value is", fields)
                    value = fields.get('type')
                    value = value[0].decode("utf-8")

                    if value == "manual":
                        print("Change to manual mode")

                    elif value == "auto":
                        print("Change to auto mode")
                    
                    elif value == "light_sensor":
                        print("apply Light sensor")

                    elif value == "no_light_sensor":
                        print("not apply Light sensor")
                    else:
                        print("unknown vlaue")

                    print("value applied is ", value)
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/text')
                    self.end_headers()
                    self.wfile.write(value.encode(encoding='utf_8'))

            if self.path.endswith("/light"):
                ctype, pdict = cgi.parse_header(self.headers['content-type'])
                pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
                if ctype == 'multipart/form-data':
                    fields = cgi.parse_multipart(self.rfile, pdict)
                    print("Fields value is", fields)
                    value = fields.get('value')
                    value = value[0].decode("utf-8")
                    print("value applied is ", value)

                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/text')
                    self.end_headers()
                    self.wfile.write(value.encode(encoding='utf_8'))

        # except:
        #     self.send_response(400)
        #     self.send_header('Content-Type', 'application/text')
        #     self.end_headers()
        #     returnVal = 'error'
        #     self.wfile.write(returnVal.encode(encoding='utf_8'))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


if __name__ == '__main__':
    server = ThreadedHTTPServer(('192.168.124.41', 8080), Handler)
    # server.socket = ssl.wrap_socket(server.socket,
    # keyfile="keys/private.key", 
    # certfile='keys/certificate.cert', server_side=True)
    print('Starting server, use <Ctrl-C> to stop')
    server.serve_forever()
