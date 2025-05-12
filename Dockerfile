FROM ruby:3.0.2

WORKDIR /app
COPY Gemfile /app/
COPY Gemfile.lock /app/

RUN bundle install

COPY . /app/

EXPOSE 4000

CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0"]