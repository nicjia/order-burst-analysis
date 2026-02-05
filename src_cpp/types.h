#ifndef TYPES_H
#define TYPES_H

#include <string>

//time --> seconds after midnight precision up to milliseconds
//event types --> 1 submission of limit order, 2 cancellation of limit order, 3 deletion of limit order
//event types --> 4 execution of limit order, 5 execution of hidden order, 6 cross trade, 7 trading halt indicator
//price --> dollar size * 10000
// Direction --> -1 sell, 1 buy, Execution of a sell (buy) limit order corresponds to a buyer (seller) initiated trade, i.e. buy (sell) trade.

struct LobsterMessage {
    double time;
    int type;       
    long order_id;
    int size;
    int price;
    int direction;  
};

#endif