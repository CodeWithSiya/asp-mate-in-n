file(1..8).
rank(1..8).

opponent(white,black).
opponent(black,white).

occupied(X,Y) :- piece(_,_,X,Y).
occupied(X,Y) :- king(_,X,Y).

own_piece(C,X,Y) :- piece(C,_,X,Y).
own_piece(C,X,Y) :- king(C,X,Y).

% ---- blocking helpers (original board) ----
blocked_h(FX,FY,TX) :- occupied(BX,FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX,FY,TX) :- occupied(BX,FY), FX > TX, BX < FX, BX > TX.

blocked_v(FX,FY,TY) :- occupied(FX,BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX,FY,TY) :- occupied(FX,BY), FY > TY, BY < FY, BY > TY.

blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), TX > FX, TY > FY, BX > FX, BX < TX, BY > FY, BY < TY, BX-FX = BY-FY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), FX > TX, TY > FY, BX < FX, BX > TX, BY > FY, BY < TY, FX-BX = BY-FY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), TX > FX, FY > TY, BX > FX, BX < TX, BY < FY, BY > TY, BX-FX = FY-BY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), FX > TX, FY > TY, BX < FX, BX > TX, BY < FY, BY > TY, FX-BX = FY-BY.

% ---- queen moves ----
queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FY = TY, FX != TX,
    not blocked_h(FX,FY,TX), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX = TX, FY != TY,
    not blocked_v(FX,FY,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), TX > FX, TY > FY, TX-FX = TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX > TX, TY > FY, FX-TX = TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), TX > FX, FY > TY, TX-FX = FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX > TX, FY > TY, FX-TX = FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

% ---- rook moves ----
rook_move(FX,FY,TX,TY) :-
    piece(C,rook,FX,FY), to_move(C),
    file(TX), rank(TY), FY = TY, FX != TX,
    not blocked_h(FX,FY,TX), not own_piece(C,TX,TY).

rook_move(FX,FY,TX,TY) :-
    piece(C,rook,FX,FY), to_move(C),
    file(TX), rank(TY), FX = TX, FY != TY,
    not blocked_v(FX,FY,TY), not own_piece(C,TX,TY).

% ---- bishop moves ----
bishop_move(FX,FY,TX,TY) :-
    piece(C,bishop,FX,FY), to_move(C),
    file(TX), rank(TY), TX > FX, TY > FY, TX-FX = TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

bishop_move(FX,FY,TX,TY) :-
    piece(C,bishop,FX,FY), to_move(C),
    file(TX), rank(TY), FX > TX, TY > FY, FX-TX = TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

bishop_move(FX,FY,TX,TY) :-
    piece(C,bishop,FX,FY), to_move(C),
    file(TX), rank(TY), TX > FX, FY > TY, TX-FX = FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

bishop_move(FX,FY,TX,TY) :-
    piece(C,bishop,FX,FY), to_move(C),
    file(TX), rank(TY), FX > TX, FY > TY, FX-TX = FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

% ---- knight moves ----
knight_delta(2,1). knight_delta(2,-1). knight_delta(-2,1). knight_delta(-2,-1).
knight_delta(1,2). knight_delta(1,-2). knight_delta(-1,2). knight_delta(-1,-2).

knight_move(FX,FY,TX,TY) :-
    piece(C,knight,FX,FY), to_move(C),
    knight_delta(DX,DY),
    TX = FX+DX, TY = FY+DY,
    file(TX), rank(TY),
    not own_piece(C,TX,TY).

% ---- pawn moves ----
pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    TX = FX, TY = FY+1, file(TX), rank(TY),
    not occupied(TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    FY = 2, TX = FX, TY = 4, file(TX), rank(TY),
    not occupied(FX,3), not occupied(FX,4).

pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    TY = FY+1, TX = FX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(white,TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    TY = FY+1, FX = TX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(white,TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    TX = FX, TY = FY-1, file(TX), rank(TY),
    not occupied(TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    FY = 7, TX = FX, TY = 5, file(TX), rank(TY),
    not occupied(FX,6), not occupied(FX,5).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    TY = FY-1, TX = FX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(black,TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    TY = FY-1, FX = TX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(black,TX,TY).

% ---- king moves ----
king_move(FX,FY,TX,TY) :-
    king(C,FX,FY), to_move(C),
    file(TX), rank(TY),
    TX-FX <= 1, FX-TX <= 1,
    TY-FY <= 1, FY-TY <= 1,
    TX != FX,
    not own_piece(C,TX,TY).

king_move(FX,FY,TX,TY) :-
    king(C,FX,FY), to_move(C),
    file(TX), rank(TY),
    TX-FX <= 1, FX-TX <= 1,
    TY-FY <= 1, FY-TY <= 1,
    TY != FY,
    not own_piece(C,TX,TY).

% ---- candidate moves ----
cand(C,queen,FX,FY,TX,TY)  :- to_move(C), queen_move(FX,FY,TX,TY).
cand(C,rook,FX,FY,TX,TY)   :- to_move(C), rook_move(FX,FY,TX,TY).
cand(C,bishop,FX,FY,TX,TY) :- to_move(C), bishop_move(FX,FY,TX,TY).
cand(C,knight,FX,FY,TX,TY) :- to_move(C), knight_move(FX,FY,TX,TY).
cand(C,pawn,FX,FY,TX,TY)   :- to_move(C), pawn_move(FX,FY,TX,TY).
cand(C,king,FX,FY,TX,TY)   :- to_move(C), king_move(FX,FY,TX,TY).

% ---- occupied after move ----
occ_after(C,PT,FX,FY,TX,TY, SX,SY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece(_,_,SX,SY),
    not (SX = FX, SY = FY),
    not (SX = TX, SY = TY).

occ_after(C,PT,FX,FY,TX,TY, SX,SY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(KC,SX,SY),
    not (KC = C, PT = king),
    not (SX = TX, SY = TY).

occ_after(C,PT,FX,FY,TX,TY, SX,SY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(C,FX,FY),
    PT = king,
    SX = TX, SY = TY.

occ_after(C,PT,FX,FY,TX,TY, TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT != king.

% ---- blocking after move ----
blk_h_after(C,PT,FX,FY,TX,TY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,SR),
    SF < TR, BX > SF, BX < TR.

blk_h_after(C,PT,FX,FY,TX,TY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,SR),
    SF > TR, BX < SF, BX > TR.

blk_v_after(C,PT,FX,FY,TX,TY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, SF,BY),
    SR < TR, BY > SR, BY < TR.

blk_v_after(C,PT,FX,FY,TX,TY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, SF,BY),
    SR > TR, BY < SR, BY > TR.

blk_d_after(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,BY),
    TF > SF, TR > SR, BX > SF, BX < TF, BY > SR, BY < TR, BX-SF = BY-SR.

blk_d_after(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,BY),
    SF > TF, TR > SR, BX < SF, BX > TF, BY > SR, BY < TR, SF-BX = BY-SR.

blk_d_after(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,BY),
    TF > SF, SR > TR, BX > SF, BX < TF, BY < SR, BY > TR, BX-SF = SR-BY.

blk_d_after(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,BY),
    SF > TF, SR > TR, BX < SF, BX > TF, BY < SR, BY > TR, SF-BX = SR-BY.

% ---- own pieces after move ----
own_after(C,PT,FX,FY,TX,TY, SX,SY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece(C,_,SX,SY),
    not (SX = FX, SY = FY).

own_after(C,PT,FX,FY,TX,TY, TX,TY) :-
    cand(C,PT,FX,FY,TX,TY).

own_after(C,PT,FX,FY,TX,TY, KX,KY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(C,KX,KY),
    PT != king.

own_after(C,PT,FX,FY,TX,TY, TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = king.

% ---- remaining pieces of mover ----
rem_piece(C,PT,FX,FY,TX,TY, PPT,PX,PY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece(C,PPT,PX,PY),
    not (PX = FX, PY = FY).

% ---- opponent king position after move ----
opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    not (TX = OKX, TY = OKY).

% ---- check detection after move ----

% moved piece gives check
in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen, TY = OKY, TX != OKX,
    not blk_h_after(C,PT,FX,FY,TX,TY, TX,TY,OKX).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen, TX = OKX, TY != OKY,
    not blk_v_after(C,PT,FX,FY,TX,TY, TX,TY,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen, OKX > TX, OKY > TY, OKX-TX = OKY-TY,
    not blk_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen, TX > OKX, OKY > TY, TX-OKX = OKY-TY,
    not blk_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen, OKX > TX, TY > OKY, OKX-TX = TY-OKY,
    not blk_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen, TX > OKX, TY > OKY, TX-OKX = TY-OKY,
    not blk_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = rook, TY = OKY, TX != OKX,
    not blk_h_after(C,PT,FX,FY,TX,TY, TX,TY,OKX).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = rook, TX = OKX, TY != OKY,
    not blk_v_after(C,PT,FX,FY,TX,TY, TX,TY,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = bishop, OKX > TX, OKY > TY, OKX-TX = OKY-TY,
    not blk_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = bishop, TX > OKX, OKY > TY, TX-OKX = OKY-TY,
    not blk_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = bishop, OKX > TX, TY > OKY, OKX-TX = TY-OKY,
    not blk_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = bishop, TX > OKX, TY > OKY, TX-OKX = TY-OKY,
    not blk_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = knight,
    knight_delta(DX,DY),
    OKX = TX+DX, OKY = TY+DY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = pawn, C = white,
    OKY = TY+1, OKX = TX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = pawn, C = white,
    OKY = TY+1, TX = OKX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = pawn, C = black,
    TY = OKY+1, OKX = TX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = pawn, C = black,
    TY = OKY+1, TX = OKX+1.

% remaining pieces give check
in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QY = OKY, QX != OKX,
    not blk_h_after(C,PT,FX,FY,TX,TY, QX,QY,OKX).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX = OKX, QY != OKY,
    not blk_v_after(C,PT,FX,FY,TX,TY, QX,QY,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    OKX > QX, OKY > QY, OKX-QX = OKY-QY,
    not blk_d_after(C,PT,FX,FY,TX,TY, QX,QY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX > OKX, OKY > QY, QX-OKX = OKY-QY,
    not blk_d_after(C,PT,FX,FY,TX,TY, QX,QY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    OKX > QX, QY > OKY, OKX-QX = QY-OKY,
    not blk_d_after(C,PT,FX,FY,TX,TY, QX,QY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX > OKX, QY > OKY, QX-OKX = QY-OKY,
    not blk_d_after(C,PT,FX,FY,TX,TY, QX,QY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, rook,RX,RY),
    RY = OKY, RX != OKX,
    not blk_h_after(C,PT,FX,FY,TX,TY, RX,RY,OKX).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, rook,RX,RY),
    RX = OKX, RY != OKY,
    not blk_v_after(C,PT,FX,FY,TX,TY, RX,RY,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    OKX > BX, OKY > BY, OKX-BX = OKY-BY,
    not blk_d_after(C,PT,FX,FY,TX,TY, BX,BY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    BX > OKX, OKY > BY, BX-OKX = OKY-BY,
    not blk_d_after(C,PT,FX,FY,TX,TY, BX,BY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    OKX > BX, BY > OKY, OKX-BX = BY-OKY,
    not blk_d_after(C,PT,FX,FY,TX,TY, BX,BY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    BX > OKX, BY > OKY, BX-OKX = BY-OKY,
    not blk_d_after(C,PT,FX,FY,TX,TY, BX,BY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, knight,NX,NY),
    knight_delta(DX,DY),
    OKX = NX+DX, OKY = NY+DY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    C = white,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    OKY = PY+1, OKX = PX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    C = white,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    OKY = PY+1, PX = OKX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    C = black,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PY = OKY+1, OKX = PX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    C = black,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PY = OKY+1, PX = OKX+1.

% king check from mover's king
in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    king(C,KX,KY),
    PT != king,
    OKX-KX <= 1, KX-OKX <= 1,
    OKY-KY <= 1, KY-OKY <= 1,
    OKX != KX.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    king(C,KX,KY),
    PT != king,
    OKX-KX <= 1, KX-OKX <= 1,
    OKY-KY <= 1, KY-OKY <= 1,
    OKY != KY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = king,
    OKX-TX <= 1, TX-OKX <= 1,
    OKY-TY <= 1, TY-OKY <= 1,
    OKX != TX.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = king,
    OKX-TX <= 1, TX-OKX <= 1,
    OKY-TY <= 1, TY-OKY <= 1,
    OKY != TY.

% captured king = check
in_check_after(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,TX,TY).

% ---- board after move + opp king moved to escape square ----
% occ_after2: occupied after move, with opp king removed from OKX,OKY
occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, SX,SY) :-
    occ_after(C,PT,FX,FY,TX,TY, SX,SY),
    not (SX = OKX, SY = OKY).

blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,SR),
    SF < TR, BX > SF, BX < TR.

blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,SR),
    SF > TR, BX < SF, BX > TR.

blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,BY),
    SR < TR, BY > SR, BY < TR.

blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,BY),
    SR > TR, BY < SR, BY > TR.

blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,BY),
    TF > SF, TR > SR, BX > SF, BX < TF, BY > SR, BY < TR, BX-SF = BY-SR.

blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,BY),
    SF > TF, TR > SR, BX < SF, BX > TF, BY > SR, BY < TR, SF-BX = BY-SR.

blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,BY),
    TF > SF, SR > TR, BX > SF, BX < TF, BY < SR, BY > TR, BX-SF = SR-BY.

blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,BY),
    SF > TF, SR > TR, BX < SF, BX > TF, BY < SR, BY > TR, SF-BX = SR-BY.

% ---- attack on escape square EX,EY ----

% by moved piece at TX,TY
atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = queen, EY = TY, EX != TX,
    not blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = queen, EX = TX, EY != TY,
    not blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = queen, EX > TX, EY > TY, EX-TX = EY-TY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = queen, TX > EX, EY > TY, TX-EX = EY-TY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = queen, EX > TX, TY > EY, EX-TX = TY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = queen, TX > EX, TY > EY, TX-EX = TY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = rook, EY = TY, EX != TX,
    not blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = rook, EX = TX, EY != TY,
    not blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = bishop, EX > TX, EY > TY, EX-TX = EY-TY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = bishop, TX > EX, EY > TY, TX-EX = EY-TY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = bishop, EX > TX, TY > EY, EX-TX = TY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = bishop, TX > EX, TY > EY, TX-EX = TY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, TX,TY,EX,EY).

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = knight,
    knight_delta(DX,DY),
    EX = TX+DX, EY = TY+DY.

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = pawn, C = white,
    EY = TY+1, EX = TX+1.

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = pawn, C = white,
    EY = TY+1, TX = EX+1.

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = pawn, C = black,
    TY = EY+1, EX = TX+1.

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = pawn, C = black,
    TY = EY+1, TX = EX+1.

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = king,
    EX-TX <= 1, TX-EX <= 1,
    EY-TY <= 1, TY-EY <= 1,
    EX != TX.

atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = king,
    EX-TX <= 1, TX-EX <= 1,
    EY-TY <= 1, TY-EY <= 1,
    EY != TY.

% by remaining pieces
atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    EY = QY, EX != QX,
    not blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY, QX,QY,EX).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    EX = QX, EY != QY,
    not blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY, QX,QY,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    EX > QX, EY > QY, EX-QX = EY-QY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, QX,QY,EX,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX > EX, EY > QY, QX-EX = EY-QY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, QX,QY,EX,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    EX > QX, QY > EY, EX-QX = QY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, QX,QY,EX,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX > EX, QY > EY, QX-EX = QY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, QX,QY,EX,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, rook,RX,RY),
    EY = RY, EX != RX,
    not blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY, RX,RY,EX).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, rook,RX,RY),
    EX = RX, EY != RY,
    not blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY, RX,RY,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    EX > BX, EY > BY, EX-BX = EY-BY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,BY,EX,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    BX > EX, EY > BY, BX-EX = EY-BY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,BY,EX,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    EX > BX, BY > EY, EX-BX = BY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,BY,EX,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    BX > EX, BY > EY, BX-EX = BY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY, BX,BY,EX,EY).

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, knight,NX,NY),
    knight_delta(DX,DY),
    EX = NX+DX, EY = NY+DY.

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    C = white,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    EY = PY+1, EX = PX+1.

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    C = white,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    EY = PY+1, PX = EX+1.

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    C = black,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PY = EY+1, EX = PX+1.

atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    C = black,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PY = EY+1, PX = EX+1.

% by mover's king (stationary)
atk_esc_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(C,KX,KY),
    PT != king,
    EX-KX <= 1, KX-EX <= 1,
    EY-KY <= 1, KY-EY <= 1,
    EX != KX.

atk_esc_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(C,KX,KY),
    PT != king,
    EX-KX <= 1, KX-EX <= 1,
    EY-KY <= 1, KY-EY <= 1,
    EY != KY.

atk_esc_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = king,
    EX-TX <= 1, TX-EX <= 1,
    EY-TY <= 1, TY-EY <= 1,
    EX != TX.

atk_esc_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = king,
    EX-TX <= 1, TX-EX <= 1,
    EY-TY <= 1, TY-EY <= 1,
    EY != TY.

% combined attack on escape square
esc_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    atk_esc_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).
esc_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    atk_esc_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).
esc_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    atk_esc_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).

% opponent own pieces after move (blocks king escape)
opp_own_after(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,_,EX,EY),
    not (EX = TX, EY = TY).

opp_own_after(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,EX,EY).

% can opponent king escape to EX,EY?
king_can_escape(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    file(EX), rank(EY),
    EX-OKX <= 1, OKX-EX <= 1,
    EY-OKY <= 1, OKY-EY <= 1,
    EX != OKX,
    not opp_own_after(C,PT,FX,FY,TX,TY, EX,EY),
    not esc_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).

king_can_escape(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    file(EX), rank(EY),
    EX-OKX <= 1, OKX-EX <= 1,
    EY-OKY <= 1, OKY-EY <= 1,
    EY != OKY,
    not opp_own_after(C,PT,FX,FY,TX,TY, EX,EY),
    not esc_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).

% checkmate
is_checkmate(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    in_check_after(C,PT,FX,FY,TX,TY),
    not king_can_escape(C,PT,FX,FY,TX,TY,_,_).

mate_move(C,PT,FX,FY,TX,TY) :-
    is_checkmate(C,PT,FX,FY,TX,TY).

#show mate_move/6.