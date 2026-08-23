# Frontend image: Vite build stage, then a static nginx server that
# reverse-proxies /api/* to the backend container (the same prefix-stripping
# behaviour as the Vite dev server).

FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80