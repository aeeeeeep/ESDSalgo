#pragma once

#include<iostream>
#include<mysql/mysql.h>
#include<list>

using namespace std;

class MysqlConn {
private:
    MYSQL mysql_conn;
    MYSQL *mysql;
    MYSQL_RES *mysql_res;
    string url;
    string ip;
public:
    MysqlConn();

    bool connect();

    bool sqlUpdate(string buf);

    bool sqlDelete(string buf);

    bool sqlAdd(string buf);

    bool sqlQuery(string buf, list <list<string>> &resList);

    ~MysqlConn();
};
