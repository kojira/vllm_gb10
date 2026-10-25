# Build stage
FROM node:22.23.4-alpine AS builder

WORKDIR /app

# Install pnpm
RUN npm install -g pnpm@11.14.0

# Copy package files
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml* ./

# Install dependencies
RUN pnpm install --no-frozen-lockfile

# Copy source
COPY . .

# Build
RUN pnpm run build

# Production stage
FROM node:22.23.4-alpine AS runtime

WORKDIR /app

RUN npm install -g pnpm@11.14.0

COPY --from=builder /app/dist ./dist
COPY --from=builder /app/server ./server
COPY --from=builder /app/package.json ./
COPY --from=builder /app/pnpm-workspace.yaml ./

# Install only production dependencies for server
RUN pnpm install --prod

EXPOSE 3000

CMD ["node", "server/index.js"]

