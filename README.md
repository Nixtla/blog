## Nixtla Blog

This is the blog for Nixtla.
By default, vercel will host the files so to fetch a post you need to call:

`https://nixtla-blog.vercel.app/posts/<fileName>.md`

for example:

`https://nixtla-blog.vercel.app/posts/anomaly_detection_performance_evaluation.md`

## Running the blog

```bash
bun i
vercel dev
```

## Development

To test out changes before submiting them live (nixtla.io) you can merge them into the `development` branch. You can see the changes there in this link https://web-git-development-nixtla-web.vercel.app/, once you are happy with those, merge into `main` branch to deploy them to nixtla.io

Build process takes about 1 minute, so be patient while waiting for your changes to be visible.
