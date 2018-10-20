var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');

var app = express();

app.use('/dist', express.static(path.join(__dirname, 'dist')));
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules')));

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/myview.html', function(req, res) {
    res.sendFile('myview.html', {"root": __dirname});
});

app.get('/ui', function(req, res) {
    res.sendFile('ui.html', {"root": __dirname});
});

app.get('/ui2', function(req, res) {
    res.sendFile('ui2.html', {"root": __dirname});
});

app.use('/', indexRouter);
app.use('/users', usersRouter);

module.exports = app;
