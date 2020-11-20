import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

capture = None


class CamHandler(BaseHTTPRequestHandler):
    print("cam handler")
    def do_GET(self):
        print("do_get")
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=--jpgboundary'
            )
            self.end_headers()
            while True:
                print("while")
                try:

                    rc, img = capture.read()
                    if not rc:
                        continue
                    img_str = cv2.imencode('.jpg', img)[1].tostring()

                    self.send_header('Content-type', 'image/jpeg')
                    self.end_headers()
                    self.wfile.write(img_str)
                    self.wfile.write(b"\r\n--jpgboundary\r\n")

                except KeyboardInterrupt:
                    self.wfile.write(b"\r\n--jpgboundary--\r\n")
                    break
                except BrokenPipeError:
                    continue
            return

        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(b'<img src="http://127.0.0.1:8080/cam.mjpg"/>')
            self.wfile.write(b'</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def main():

    global capture
    capture = cv2.VideoCapture(0)
    global img
    try:
        server = ThreadedHTTPServer(('localhost', 8080), CamHandler)
        print("server started at http://127.0.0.1:8080/cam.html")
        server.serve_forever()
        print("serve forever!")
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()

if __name__ == '__main__':
    main()
