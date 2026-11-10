import os
import http.server
import socketserver
import webbrowser


def generate_html(glb_path, output_html):
    # Define an HTML template with a placeholder for the GLB file path.
    html_template = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>GLB Model Viewer</title>
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  </head>
  <body style="margin: 0;">
    <model-viewer
      src="{glb_file}"
      alt="3D Model"
      camera-controls
      style="width: 100vw; height: 100vh;">
    </model-viewer>
  </body>
</html>
    """
    # Replace the placeholder with the actual GLB file path.
    html_content = html_template.format(
        glb_file=os.path.relpath(glb_path, os.path.dirname(output_html))
    )
    with open(output_html, "w") as f:
        f.write(html_content)
    print(f"HTML file generated at: {output_html}")


def start_http_server(
    directory,
):
    # Change the working directory to the one where your HTML file resides.
    os.chdir(directory)

    with socketserver.TCPServer(("", 0), http.server.SimpleHTTPRequestHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://localhost:{port}/glb_viewer.html"
        print("Serving at:", url)
        webbrowser.open(url)
        httpd.serve_forever()
