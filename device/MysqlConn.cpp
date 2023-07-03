#include"MysqlConn.h"

using namespace std;

MysqlConn::MysqlConn() {
    mysql = mysql_init(&mysql_conn);
    if (mysql == NULL) {
        printf("init error\n");
        exit(0);
    }
}

bool MysqlConn::connect() {
    mysql = mysql_real_connect(mysql, "192.168.10.105", "root", "020506", "db_esds", 3307, NULL, 0);
    if (mysql == NULL) {
        printf("connect error:%s\n", mysql_error(&mysql_conn));
        return 1;
    }
    return 0;
}

bool MysqlConn::sqlAdd(string buf) {
    if (mysql_query(mysql, buf.c_str())) {
        printf("\nmysql_query error!\n");
        return 1;
    }
    return 0;
}

bool MysqlConn::sqlDelete(string buf) {
    if (mysql_query(mysql, buf.c_str())) {
        printf("\nmysql_query error!\n");
        return 1;
    }
    return 0;
}

bool MysqlConn::sqlQuery(string buf, list <list<string>> &resList) {
    if (mysql_query(mysql, buf.c_str())) {
        printf("\nmysql_query error!\n");
        return 1;
    } else {
        mysql_res = mysql_store_result(mysql);
        if (mysql_res == NULL) {
            printf("提取数据失败:%s\n", mysql_error(mysql));
            return 1;
        } else {
            int num = mysql_num_rows(mysql_res);//获取结果集中有多少行
            if (num == 0) {
                resList.clear();
                return 0;
            } else {
                MYSQL_ROW row;
                while ((row = mysql_fetch_row(mysql_res))) {
                    list <string> list1;
                    for (int t = 0; t < mysql_num_fields(mysql_res); t++) {
                        list1.push_back(row[t]);
                    }
                    resList.push_back(list1);
                }
            }
        }
    }
    return 0;
}

bool MysqlConn::sqlUpdate(string buf) {
    if (mysql_query(mysql, buf.c_str())) {
        printf("\nmysql_query error!\n");
        return 1;
    }
    return 0;
}

MysqlConn::~MysqlConn() {
    mysql_free_result(mysql_res);    //释放结果集    mysql_close(&mysql_conn);
}
