import bottle

def check_fields(data, fields):
    if not data:
        return False
    for f in fields:
        if f not in data:
            return False
    return True

def get_limit_and_offset_from_request():
    """
    Extract the limit and offset value from request
    which is help to design the pagination
    :return: limit, offset
    """

    try:
        limit = bottle.request.query.get('limit')
        offset = bottle.request.query.get('offset')
        if not limit:
            limit = 5
        if not offset:
            offset = 0
        # handle if limit in string like ss
        limit = int(limit)
        offset = int(offset)
    except Exception as ex:
        logging.exception('Query param parsing issue :(%s)' %(ex))
        limit = 5
        offset = 0

    return limit, offset

