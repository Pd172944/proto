`app.log` is a web server log. Each line looks like:

    2024-03-11T09:14:02Z ERROR db  connection refused
    2024-03-11T09:14:05Z INFO  http GET /health 200

Write a file `errors.txt` in the project root containing every `ERROR` line from
`app.log`, in the order it appears, with the lines reproduced exactly as they are in
the log. Nothing else should be in the file.
