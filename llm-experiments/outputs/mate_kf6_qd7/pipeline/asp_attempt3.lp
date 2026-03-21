file(1..8).
rank(1..8).

occupied(X,Y) :- piece(_,_,X,Y).
occupied(X,Y) :- king(_,X,Y).

own_white(X,Y) :- piece(white,_,X,Y).
own_white(X,Y) :- king(white,X,Y).

diff_le1(0).
diff_le1(1).

% Adjacent squares helper
adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X2 - X1, DX >= 0, diff_le1(DX),
    DY = Y2 - Y1, DY >= 0, diff_le1(DY),
    X1 = X2, Y1 != Y2.

adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X1 - X2, DX >= 0, diff_le1(DX),
    DY = Y2 - Y1, DY >= 0, diff_le1(DY),
    X1 = X2, Y1 != Y2.

adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X2 - X1, DX > 0, diff_le1(DX),
    DY = Y2 - Y1, DY >= 0, diff_le1(DY).

adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X1 - X2, DX > 0, diff_le1(DX),
    DY = Y2 - Y1, DY >= 0, diff_le1(DY).

adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X2 - X1, DX > 0, diff_le1(DX),
    DY = Y1 - Y2, DY > 0, diff_le1(DY).

adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X1 - X2, DX > 0, diff_le1(DX),
    DY = Y1 - Y2, DY > 0, diff_le1(DY).

% Queen move types
queen_move_h(FX,FY,TX,FY) :-
    piece(white,queen,FX,FY), file(TX), TX != FX.

queen_move_v(FX,FY,FX,TY) :-
    piece(white,queen,FX,FY), rank(TY), TY != FY.

queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX != FX,
    D = TX - FX, D > 0, D2 = TY - FY, D2 > 0, D = D2.

queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX != FX,
    D = FX - TX, D > 0, D2 = TY - FY, D2 > 0, D = D2.

queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX != FX,
    D = TX - FX, D > 0, D2 = FY - TY, D2 > 0, D = D2.

queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX != FX,
    D = FX - TX, D > 0, D2 = FY - TY, D2 > 0, D = D2.

% Blocking
blocked_h(FX,FY,TX) :-
    occupied(BX,FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX,FY,TX) :-
    occupied(BX,FY), FX > TX, BX < FX, BX > TX.

blocked_v(FX,FY,TY) :-
    occupied(FX,BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX,FY,TY) :-
    occupied(FX,BY), FY > TY, BY < FY, BY > TY.

blocked_d(FX,FY,TX,TY) :-
    occupied(BX,BY), FX < TX, FY < TY,
    BX > FX, BX < TX, BY > FY, BY < TY,
    BX - FX = BY - FY.

blocked_d(FX,FY,TX,TY) :-
    occupied(BX,BY), FX > TX, FY < TY,
    BX < FX, BX > TX, BY > FY, BY < TY,
    FX - BX = BY - FY.

blocked_d(FX,FY,TX,TY) :-
    occupied(BX,BY), FX < TX, FY > TY,
    BX > FX, BX < TX, BY < FY, BY > TY,
    BX - FX = FY - BY.

blocked_d(FX,FY,TX,TY) :-
    occupied(BX,BY), FX > TX, FY > TY,
    BX < FX, BX > TX, BY < FY, BY > TY,
    FX - BX = FY - BY.

% Valid queen moves
queen_can_move(FX,FY,TX,TY) :-
    queen_move_h(FX,FY,TX,TY),
    not blocked_h(FX,FY,TX),
    not own_white(TX,TY).

queen_can_move(FX,FY,TX,TY) :-
    queen_move_v(FX,FY,TX,TY),
    not blocked_v(FX,FY,TY),
    not own_white(TX,TY).

queen_can_move(FX,FY,TX,TY) :-
    queen_move_d(FX,FY,TX,TY),
    not blocked_d(FX,FY,TX,TY),
    not own_white(TX,TY).

% Occupied after queen move from (FX,FY) to (QX,QY)
occ_after(BX,BY,FX,FY,QX,QY) :-
    occupied(BX,BY), queen_can_move(FX,FY,QX,QY),
    not same_as(BX,BY,FX,FY).

occ_after(QX,QY,FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY).

same_as(X,Y,X,Y) :- occupied(X,Y).

% Blocking after move
blk_h_after(QX,QY,AX,FX,FY) :-
    occ_after(BX,QY,FX,FY,QX,QY),
    QX < AX, BX > QX, BX < AX.
blk_h_after(QX,QY,AX,FX,FY) :-
    occ_after(BX,QY,FX,FY,QX,QY),
    QX > AX, BX < QX, BX > AX.

blk_v_after(QX,QY,AY,FX,FY) :-
    occ_after(QX,BY,FX,FY,QX,QY),
    QY < AY, BY > QY, BY < AY.
blk_v_after(QX,QY,AY,FX,FY) :-
    occ_after(QX,BY,FX,FY,QX,QY),
    QY > AY, BY < QY, BY > AY.

blk_d_after(QX,QY,AX,AY,FX,FY) :-
    occ_after(BX,BY,FX,FY,QX,QY),
    QX < AX, QY < AY, BX > QX, BX < AX, BY > QY, BY < AY,
    BX - QX = BY - QY.

blk_d_after(QX,QY,AX,AY,FX,FY) :-
    occ_after(BX,BY,FX,FY,QX,QY),
    QX > AX, QY < AY, BX < QX, BX > AX, BY > QY, BY < AY,
    QX - BX = BY - QY.

blk_d_after(QX,QY,AX,AY,FX,FY) :-
    occ_after(BX,BY,FX,FY,QX,QY),
    QX < AX, QY > AY, BX > QX, BX < AX, BY < QY, BY > AY,
    BX - QX = QY - BY.

blk_d_after(QX,QY,AX,AY,FX,FY) :-
    occ_after(BX,BY,FX,FY,QX,QY),
    QX > AX, QY > AY, BX < QX, BX > AX, BY < QY, BY > AY,
    QX - BX = QY - BY.

% Queen at (QX,QY) attacks (AX,AY) after move from (FX,FY)
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AY = QY, AX != QX,
    not blk_h_after(QX,QY,AX,FX,FY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX = QX, AY != QY,
    not blk_v_after(QX,QY,AY,FX,FY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX != QX,
    D = AX - QX, D > 0, D2 = AY - QY, D2 > 0, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX != QX,
    D = QX - AX, D > 0, D2 = AY - QY, D2 > 0, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX != QX,
    D = AX - QX, D > 0, D2 = QY - AY, D2 > 0, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX != QX,
    D = QX - AX, D > 0, D2 = QY - AY, D2 > 0, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY).

% White king attacks
wk_att(AX,AY) :-
    king(white,KX,KY), adj(KX,KY,AX,AY).

% White attacks after queen move
white_att_after(AX,AY,FX,FY,QX,QY) :-
    q_att_after(QX,QY,AX,AY,FX,FY).

white_att_after(AX,AY,FX,FY,QX,QY) :-
    wk_att(AX,AY), queen_can_move(FX,FY,QX,QY).

% White pieces on board after move
white_occ_after(X,Y,FX,FY,QX,QY) :-
    piece(white,_,X,Y), queen_can_move(FX,FY,QX,QY),
    not same_as(X,Y,FX,FY).

white_occ_after(QX,QY,FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY).

white_occ_after(X,Y,FX,FY,QX,QY) :-
    king(white,X,Y), queen_can_move(FX,FY,QX,QY).

% Queen defended by white king after move
queen_def_after(QX,QY) :- wk_att(QX,QY).

% Black king adjacent squares
bk_adj(AX,AY) :-
    king(black,KX,KY), adj(KX,KY,AX,AY).

% Black king can escape
bk_escape(FX,FY,QX,QY) :-
    bk_adj(AX,AY),
    queen_can_move(FX,FY,QX,QY),
    not white_att_after(AX,AY,FX,FY,QX,QY),
    not white_occ_after(AX,AY,FX,FY,QX,QY).

% Black king captures undefended queen
bk_escape(FX,FY,QX,QY) :-
    bk_adj(QX,QY),
    queen_can_move(FX,FY,QX,QY),
    not queen_def_after(QX,QY),
    not white_att_after(QX,QY,FX,FY,QX,QY).

% In check after move
in_check_after(FX,FY,QX,QY) :-
    king(black,KX,KY),
    white_att_after(KX,KY,FX,FY,QX,QY).

% Checkmate
is_checkmate(FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY),
    in_check_after(FX,FY,QX,QY),
    not bk_escape(FX,FY,QX,QY).

mate_move(white,queen,FX,FY,QX,QY) :-
    is_checkmate(FX,FY,QX,QY).

#show mate_move/6.