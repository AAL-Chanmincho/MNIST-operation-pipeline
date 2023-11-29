# Dockerfile-fe
FROM node:20

WORKDIR /app

COPY . /app

RUN npm install
RUN npm run build

EXPOSE 4321

CMD ["npm", "start"]
