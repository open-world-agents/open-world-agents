# Stage 1: Build
FROM node:20-alpine AS builder

RUN apk add --no-cache git

ARG CACHE_BUST=1
RUN git clone --depth 1 https://github.com/open-world-agents/open-world-agents /owa

WORKDIR /owa/projects/owa-dataset-visualizer
RUN npm ci && npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

COPY --from=builder /owa/projects/owa-dataset-visualizer/dist /usr/share/nginx/html
COPY --from=builder /owa/projects/owa-dataset-visualizer/nginx.conf /etc/nginx/conf.d/default.conf

# HuggingFace Spaces uses port 7860
EXPOSE 7860
CMD ["nginx", "-g", "daemon off;"]
