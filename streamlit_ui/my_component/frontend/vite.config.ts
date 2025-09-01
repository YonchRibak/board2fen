import { defineConfig, loadEnv, UserConfig } from "vite"

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())
  const port = env.VITE_PORT ? parseInt(env.VITE_PORT) : 3002

  return {
    base: "./",
    server: {
      host: "127.0.0.1",
      port,
      cors: true,
      hmr: {
        host: "127.0.0.1",
        port,
        protocol: "ws",
      },
    },
    build: {
      outDir: "build",
    },
  } satisfies UserConfig
})
