import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)));
const port = Number(process.env.PORT || 4173);

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".gif": "image/gif",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
};

function resolveRequestPath(requestUrl) {
  const pathname = decodeURIComponent(new URL(requestUrl, "http://127.0.0.1").pathname);
  const candidate = resolve(root, "." + normalize(pathname));
  const relativePath = relative(root, candidate);
  if (relativePath.startsWith("..") || relativePath.includes(".." + "/")) {
    return null;
  }
  return candidate;
}

async function findFile(requestUrl) {
  const candidate = resolveRequestPath(requestUrl);
  if (!candidate) return null;

  try {
    const candidateStats = await stat(candidate);
    if (candidateStats.isFile()) return candidate;
  } catch {
    // Client-side routes fall through to index.html below.
  }

  if (!extname(candidate)) return join(root, "index.html");
  return null;
}

const server = createServer(async (request, response) => {
  if (request.method !== "GET" && request.method !== "HEAD") {
    response.writeHead(405, { Allow: "GET, HEAD" });
    response.end("Method not allowed");
    return;
  }

  const filePath = await findFile(request.url || "/");
  if (!filePath) {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not found");
    return;
  }

  try {
    const content = await readFile(filePath);
    response.writeHead(200, {
      "Cache-Control": "no-cache",
      "Content-Length": content.byteLength,
      "Content-Type": mimeTypes[extname(filePath).toLowerCase()] || "application/octet-stream",
    });
    if (request.method === "HEAD") {
      response.end();
    } else {
      response.end(content);
    }
  } catch {
    response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Unable to read local demo file");
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log("Reflexion marketing demo running at http://127.0.0.1:" + port);
});
