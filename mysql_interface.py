

def get_connection():
    import pymysql.cursors
    con = pymysql.connect(host='localhost',
                                 user='****',
                                 password='******',
                                 db='myoquant',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
    return con

def add_fibers(mousename, fibers):
    con = get_connection()
    try:
        with con.cursor() as cursor:
            cursor.execute("SELECT name FROM `mice`")
            result = cursor.fetchall()
        names = [r['name'] for r in result]
        if mousename in names:
            return "An entry with the mousname '{}' already exists. Change the name and try again".format(mousename)
        with con.cursor() as cursor:
            sql = "INSERT INTO `mice` (`name`) VALUES (%s);"
            cursor.execute(sql, (mousename))
        con.commit()
        with con.cursor() as cursor:
            sql = """
                INSERT INTO `fibers` SET 
                  `mouse_id` = (SELECT `id` FROM `mice` WHERE `name`=%s), 
                  `area` = %s,
                  `eccentricity` = %s,
                  `convexity` = %s,
                  `circularity` = %s,
                  `minor_axis_length` = %s;
            """
            for f in fibers:
                f2 = [float(n) for n in f]
                cursor.execute(sql, (mousename, f2[0], f2[1], f2[2], f2[3], f2[5]))
        con.commit()
    finally:
        con.close()
    return 'Successfully added all fibers to database'