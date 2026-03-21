file(1..8).
rank(1..8).

occupied(X,Y) :- piece(_,_,X,Y).
occupied(X,Y) :- king(_,X,Y).

own_white(X,Y) :- piece(white,_,X,Y).
own_white(X,Y) :- king(white,X,Y).

% Adjacent squares: all squares within 1 step
adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X2 - X1, DX >= 0, DX <= 1,
    DY = Y2 - Y1, DY >= 0, DY <= 1,
    DX + DY > 0.

adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X1 - X2, DX > 0, DX <= 1,
    DY = Y2 - Y1, DY >= 0, DY <= 1.

adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X2 - X1, DX >= 0, DX <= 1,
    DY = Y1 - Y2, DY > 0, DY <= 1.

adj(X1,Y1,X2,Y2) :-
    file(X1), rank(Y1), file(X2), rank(Y2),
    DX = X1 - X2, DX > 0, DX <= 1,
    DY = Y1 - Y2, DY > 0, DY <= 1.

% Queen move candidates (horizontal)
queen_move_h(FX,FY,TX,FY) :-
    piece(white,queen,FX,FY), file(TX), TX > FX.
queen_move_h(FX,FY,TX,FY) :-
    piece(white,queen,FX,FY), file(TX), TX < FX.

% Queen move candidates (vertical)
queen_move_v(FX,FY,FX,TY) :-
    piece(white,queen,FX,FY), rank(TY), TY > FY.
queen_move_v(FX,FY,FX,TY) :-
    piece(white,queen,FX,FY), rank(TY), TY < FY.

% Queen move candidates (diagonal)
queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    D = TX - FX, D > 0, D2 = TY - FY, D2 > 0, D = D2.
queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    D = FX - TX, D > 0, D2 = TY - FY, D2 > 0, D = D2.
queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    D = TX - FX, D > 0, D2 = FY - TY, D2 > 0, D = D2.
queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    D = FX - TX, D > 0, D2 = FY - TY, D2 > 0, D = D2.

% Blocking for horizontal moves
blocked_h(FX,FY,TX) :-
    piece(white,queen,FX,FY), file(TX), TX > FX,
    occupied(BX,FY), BX > FX, BX < TX.
blocked_h(FX,FY,TX) :-
    piece(white,queen,FX,FY), file(TX), TX < FX,
    occupied(BX,FY), BX < FX, BX > TX.

% Blocking for vertical moves
blocked_v(FX,FY,TY) :-
    piece(white,queen,FX,FY), rank(TY), TY > FY,
    occupied(FX,BY), BY > FY, BY < TY.
blocked_v(FX,FY,TY) :-
    piece(white,queen,FX,FY), rank(TY), TY < FY,
    occupied(FX,BY), BY < FY, BY > TY.

% Blocking for diagonal moves
blocked_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX > FX, TY > FY,
    occupied(BX,BY), BX > FX, BX < TX, BY > FY, BY < TY,
    BX - FX = BY - FY.
blocked_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX < FX, TY > FY,
    occupied(BX,BY), BX < FX, BX > TX, BY > FY, BY < TY,
    FX - BX = BY - FY.
blocked_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX > FX, TY < FY,
    occupied(BX,BY), BX > FX, BX < TX, BY < FY, BY > TY,
    BX - FX = FY - BY.
blocked_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX < FX, TY < FY,
    occupied(BX,BY), BX < FX, BX > TX, BY < FY, BY > TY,
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
% A square is occupied after the move if it was occupied before (except FX,FY) or is QX,QY
is_from(X,Y,FX,FY) :- queen_can_move(FX,FY,_,_), X = FX, Y = FY.

occ_after(BX,BY,FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY),
    occupied(BX,BY),
    not is_from(BX,BY,FX,FY).
occ_after(QX,QY,FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY).

% Blocking after move (horizontal)
blk_h_after(QX,QY,AX,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), AX > QX,
    occ_after(BX,QY,FX,FY,QX,QY), BX > QX, BX < AX.
blk_h_after(QX,QY,AX,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), AX < QX,
    occ_after(BX,QY,FX,FY,QX,QY), BX < QX, BX > AX.

% Blocking after move (vertical)
blk_v_after(QX,QY,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    rank(AY), AY > QY,
    occ_after(QX,BY,FX,FY,QX,QY), BY > QY, BY < AY.
blk_v_after(QX,QY,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    rank(AY), AY < QY,
    occ_after(QX,BY,FX,FY,QX,QY), BY < QY, BY > AY.

% Blocking after move (diagonal)
blk_d_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX > QX, AY > QY,
    occ_after(BX,BY,FX,FY,QX,QY),
    BX > QX, BX < AX, BY > QY, BY < AY,
    BX - QX = BY - QY.
blk_d_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX < QX, AY > QY,
    occ_after(BX,BY,FX,FY,QX,QY),
    BX < QX, BX > AX, BY > QY, BY < AY,
    QX - BX = BY - QY.
blk_d_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX > QX, AY < QY,
    occ_after(BX,BY,FX,FY,QX,QY),
    BX > QX, BX < AX, BY < QY, BY > AY,
    BX - QX = QY - BY.
blk_d_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX < QX, AY < QY,
    occ_after(BX,BY,FX,FY,QX,QY),
    BX < QX, BX > AX, BY < QY, BY > AY,
    QX - BX = QY - BY.

% Queen at (QX,QY) attacks (AX,AY) after move from (FX,FY)
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AY = QY, AX > QX,
    not blk_h_after(QX,QY,AX,FX,FY).
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AY = QY, AX < QX,
    not blk_h_after(QX,QY,AX,FX,FY).
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX = QX, AY > QY,
    not blk_v_after(QX,QY,AY,FX,FY).
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX = QX, AY < QY,
    not blk_v_after(QX,QY,AY,FX,FY).
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX > QX, AY > QY,
    D = AX - QX, D2 = AY - QY, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY).
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX < QX, AY > QY,
    D = QX - AX, D2 = AY - QY, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY).
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX > QX, AY < QY,
    D = AX - QX, D2 = QY - AY, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY).
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX < QX, AY < QY,
    D = QX - AX, D2 = QY - AY, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY).

% White king attacks
wk_att(AX,AY) :-
    king(white,KX,KY), adj(KX,KY,AX,AY),
    file(AX), rank(AY).

% White attacks after queen move
white_att_after(AX,AY,FX,FY,QX,QY) :-
    q_att_after(QX,QY,AX,AY,FX,FY).
white_att_after(AX,AY,FX,FY,QX,QY) :-
    wk_att(AX,AY), queen_can_move(FX,FY,QX,QY).

% White occupied after move
white_occ_after(X,Y,FX,FY,QX,QY) :-
    piece(white,_,X,Y), queen_can_move(FX,FY,QX,QY),
    not is_from(X,Y,FX,FY).
white_occ_after(QX,QY,FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY).
white_occ_after(X,Y,FX,FY,QX,QY) :-
    king(white,X,Y), queen_can_move(FX,FY,QX,QY).

% Queen defended by white king after move
queen_def_after(QX,QY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    wk_att(QX,QY).

% Black king adjacent squares
bk_adj(AX,AY) :-
    king(black,KX,KY), adj(KX,KY,AX,AY),
    file(AX), rank(AY).

% Black king can escape to adjacent square
bk_escape(FX,FY,QX,QY) :-
    bk_adj(AX,AY),
    queen_can_move(FX,FY,QX,QY),
    not white_att_after(AX,AY,FX,FY,QX,QY),
    not white_occ_after(AX,AY,FX,FY,QX,QY).

% Black king captures queen if undefended and not still in check
bk_escape(FX,FY,QX,QY) :-
    bk_adj(QX,QY),
    queen_can_move(FX,FY,QX,QY),
    not queen_def_after(QX,QY,FX,FY).

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