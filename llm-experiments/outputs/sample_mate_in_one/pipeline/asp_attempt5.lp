file(1..8).
rank(1..8).

opponent(white,black).
opponent(black,white).

occupied(X,Y) :- piece(_,_,X,Y).
occupied(X,Y) :- king(_,X,Y).

own_piece(C,X,Y) :- piece(C,_,X,Y).
own_piece(C,X,Y) :- king(C,X,Y).

blocked_h(FX,FY,TX) :- occupied(BX,FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX,FY,TX) :- occupied(BX,FY), FX > TX, BX < FX, BX > TX.

blocked_v(FX,FY,TY) :- occupied(FX,BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX,FY,TY) :- occupied(FX,BY), FY > TY, BY < FY, BY > TY.

blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), TX > FX, TY > FY, BX > FX, BX < TX, BY > FY, BY < TY, BX-FX = BY-FY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), FX > TX, TY > FY, BX < FX, BX > TX, BY > FY, BY < TY, FX-BX = BY-FY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), TX > FX, FY > TY, BX > FX, BX < TX, BY < FY, BY > TY, BX-FX = FY-BY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), FX > TX, FY > TY, BX < FX, BX > TX, BY < FY, BY > TY, FX-BX = FY-BY.

queen_can_move_h(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FY = TY, TX != FX,
    not blocked_h(FX,FY,TX), not own_piece(C,TX,TY).

queen_can_move_v(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX = TX, TY != FY,
    not blocked_v(FX,FY,TY), not own_piece(C,TX,TY).

queen_can_move_d1(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), TX > FX, TY > FY, TX-FX = TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_can_move_d2(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX > TX, TY > FY, FX-TX = TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_can_move_d3(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), TX > FX, FY > TY, TX-FX = FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_can_move_d4(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX > TX, FY > TY, FX-TX = FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :- queen_can_move_h(FX,FY,TX,TY).
queen_move(FX,FY,TX,TY) :- queen_can_move_v(FX,FY,TX,TY).
queen_move(FX,FY,TX,TY) :- queen_can_move_d1(FX,FY,TX,TY).
queen_move(FX,FY,TX,TY) :- queen_can_move_d2(FX,FY,TX,TY).
queen_move(FX,FY,TX,TY) :- queen_can_move_d3(FX,FY,TX,TY).
queen_move(FX,FY,TX,TY) :- queen_can_move_d4(FX,FY,TX,TY).

rook_move(FX,FY,TX,TY) :-
    piece(C,rook,FX,FY), to_move(C),
    file(TX), rank(TY), FY = TY, TX != FX,
    not blocked_h(FX,FY,TX), not own_piece(C,TX,TY).

rook_move(FX,FY,TX,TY) :-
    piece(C,rook,FX,FY), to_move(C),
    file(TX), rank(TY), FX = TX, TY != FY,
    not blocked_v(FX,FY,TY), not own_piece(C,TX,TY).

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

knight_delta(2,1). knight_delta(2,-1). knight_delta(-2,1). knight_delta(-2,-1).
knight_delta(1,2). knight_delta(1,-2). knight_delta(-1,2). knight_delta(-1,-2).

knight_move(FX,FY,TX,TY) :-
    piece(C,knight,FX,FY), to_move(C),
    knight_delta(DX,DY),
    TX = FX+DX, TY = FY+DY,
    file(TX), rank(TY),
    not own_piece(C,TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    TX = FX, TY = FY+1, file(TX), rank(TY),
    not occupied(TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    FY = 2, TX = FX, TY = 4,
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
    FY = 7, TX = FX, TY = 5,
    not occupied(FX,6), not occupied(FX,5).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    TY = FY-1, TX = FX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(black,TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    TY = FY-1, FX = TX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(black,TX,TY).

king_delta(-1,-1). king_delta(-1,0). king_delta(-1,1).
king_delta(0,-1).                    king_delta(0,1).
king_delta(1,-1).  king_delta(1,0).  king_delta(1,1).

king_move(FX,FY,TX,TY) :-
    king(C,FX,FY), to_move(C),
    king_delta(DX,DY),
    TX = FX+DX, TY = FY+DY,
    file(TX), rank(TY),
    not own_piece(C,TX,TY).

cand(C,queen,FX,FY,TX,TY)  :- to_move(C), queen_move(FX,FY,TX,TY), piece(C,queen,FX,FY).
cand(C,rook,FX,FY,TX,TY)   :- to_move(C), rook_move(FX,FY,TX,TY), piece(C,rook,FX,FY).
cand(C,bishop,FX,FY,TX,TY) :- to_move(C), bishop_move(FX,FY,TX,TY), piece(C,bishop,FX,FY).
cand(C,knight,FX,FY,TX,TY) :- to_move(C), knight_move(FX,FY,TX,TY), piece(C,knight,FX,FY).
cand(C,pawn,FX,FY,TX,TY)   :- to_move(C), pawn_move(FX,FY,TX,TY), piece(C,pawn,FX,FY).
cand(C,king,FX,FY,TX,TY)   :- to_move(C), king_move(FX,FY,TX,TY), king(C,FX,FY).

same_sq(X,Y,X,Y).

not_from(FX,FY,TX,TY,SX,SY) :- cand(_,_,FX,FY,TX,TY), file(SX), rank(SY), SX != FX.
not_from(FX,FY,TX,TY,SX,SY) :- cand(_,_,FX,FY,TX,TY), file(SX), rank(SY), SY != FY.

is_from_sq(FX,FY,TX,TY,FX,FY) :- cand(_,_,FX,FY,TX,TY).
is_to_sq(FX,FY,TX,TY,TX,TY)   :- cand(_,_,FX,FY,TX,TY).

occ_after(C,PT,FX,FY,TX,TY, SX,SY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece(_,_,SX,SY),
    not is_from_sq(FX,FY,TX,TY,SX,SY),
    not is_to_sq(FX,FY,TX,TY,SX,SY).

occ_after(C,PT,FX,FY,TX,TY, SX,SY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(KC,SX,SY),
    not is_from_sq(FX,FY,TX,TY,SX,SY),
    not is_to_sq(FX,FY,TX,TY,SX,SY),
    KC = KC.

occ_after(C,PT,FX,FY,TX,TY, TX,TY) :- cand(C,PT,FX,FY,TX,TY).

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

attacks_h_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PY = EY, PX != EX,
    not blk_h_after(C,PT,FX,FY,TX,TY, PX,PY,EX).

attacks_v_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PX = EX, PY != EY,
    not blk_v_after(C,PT,FX,FY,TX,TY, PX,PY,EY).

attacks_d_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    EX > PX, EY > PY, EX-PX = EY-PY,
    not blk_d_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY).

attacks_d_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PX > EX, EY > PY, PX-EX = EY-PY,
    not blk_d_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY).

attacks_d_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    EX > PX, PY > EY, EX-PX = PY-EY,
    not blk_d_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY).

attacks_d_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PX > EX, PY > EY, PX-EX = PY-EY,
    not blk_d_after(C,PT,FX,FY,TX,TY, PX,PY,EX,EY).

mover_piece_after(C,PT,FX,FY,TX,TY, PPT,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY), PPT = PT.

mover_piece_after(C,PT,FX,FY,TX,TY, PPT,PX,PY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece(C,PPT,PX,PY),
    PX != FX.

mover_piece_after(C,PT,FX,FY,TX,TY, PPT,PX,PY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece(C,PPT,PX,PY),
    PY != FY.

mover_king_after(C,PT,FX,FY,TX,TY, TX,TY) :-
    cand(C,PT,FX,FY,TX,TY), PT = king.

mover_king_after(C,PT,FX,FY,TX,TY, KX,KY) :-
    cand(C,PT,FX,FY,TX,TY), PT != king, king(C,KX,KY).

opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    OKX != TX.

opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    OKY != TY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen,
    attacks_h_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen,
    attacks_v_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = queen,
    attacks_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = rook,
    attacks_h_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = rook,
    attacks_v_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = bishop,
    attacks_d_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

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

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = king,
    king_delta(DX,DY),
    OKX = TX+DX, OKY = TY+DY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,TX,TY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    mover_piece_after(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX != TX,
    attacks_h_after(C,PT,FX,FY,TX,TY, QX,QY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    mover_piece_after(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX != TX,
    attacks_v_after(C,PT,FX,FY,TX,TY, QX,QY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    mover_piece_after(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX != TX,
    attacks_d_after(C,PT,FX,FY,TX,TY, QX,QY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    mover_piece_after(C,PT,FX,FY,TX,TY, rook,RX,RY),
    RX != TX,
    attacks_h_after(C,PT,FX,FY,TX,TY, RX,RY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    mover_piece_after(C,PT,FX,FY,TX,TY, rook,RX,RY),
    RX != TX,
    attacks_v_after(C,PT,FX,FY,TX,TY, RX,RY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    mover_piece_after(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    BX != TX,
    attacks_d_after(C,PT,FX,FY,TX,TY, BX,BY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    mover_piece_after(C,PT,FX,FY,TX,TY, knight,NX,NY),
    NX != TX,
    knight_delta(DX,DY),
    OKX = NX+DX, OKY = NY+DY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    C = white,
    mover_piece_after(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PX != TX,
    OKY = PY+1, OKX = PX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    C = white,
    mover_piece_after(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PX != TX,
    OKY = PY+1, PX = OKX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    C = black,
    mover_piece_after(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PX != TX,
    PY = OKY+1, OKX = PX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    C = black,
    mover_piece_after(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PX != TX,
    PY = OKY+1, PX = OKX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_pos(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT != king,
    mover_king_after(C,PT,FX,FY,TX,TY, KX,KY),
    king_delta(DX,DY),
    OKX = KX+DX, OKY = KY+DY.

occ_after_no_opp_king(C,PT,FX,FY,TX,TY, SX,SY) :-
    occ_after(C,PT,FX,FY,TX,TY, SX,SY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    SX != OKX.

occ_after_no_opp_king(C,PT,FX,FY,TX,TY, SX,SY) :-
    occ_after(C,PT,FX,FY,TX,TY, SX,SY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    SY != OKY.

blk_h_esc(C,PT,FX,FY,TX,TY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after_no_opp_king(C,PT,FX,FY,TX,TY, BX,SR),
    SF < TR, BX > SF, BX < TR.

blk_h_esc(C,PT,FX,FY,TX,TY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after_no_opp_king(C,PT,FX,FY,TX,TY, BX,SR),
    SF > TR, BX < SF, BX > TR.

blk_v_esc(C,PT,FX,FY,TX,TY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after_no_opp_king(C,PT,FX,FY,TX,TY, SF,BY),
    SR < TR, BY > SR, BY < TR.

blk_v_esc(C,PT,FX,FY,TX,TY, SF,SR,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after_no_opp_king(C,PT,FX,FY,TX,TY, SF,BY),
    SR > TR, BY < SR, BY > TR.

blk_d_esc(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after_no_opp_king(C,PT,FX,FY,TX,TY, BX,BY),
    TF > SF, TR > SR, BX > SF, BX < TF, BY > SR, BY < TR, BX-SF = BY-SR.

blk_d_esc(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after_no_opp_king(C,PT,FX,FY,TX,TY, BX,BY),
    SF > TF, TR > SR, BX < SF, BX > TF, BY > SR, BY < TR, SF-BX = BY-SR.

blk_d_esc(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after_no_opp_king(C,PT,FX,FY,TX,TY, BX,BY),
    TF > SF, SR > TR, BX > SF, BX < TF, BY < SR, BY > TR, BX-SF = SR-BY.

blk_d_esc(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after_no_opp_king(C,PT,FX,FY,TX,TY, BX,BY),
    SF > TF, SR > TR, BX < SF, BX > TF, BY < SR, BY > TR, SF-BX = SR-BY.

atk_esc_h(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PY = EY, PX != EX,
    not blk_h_esc(C,PT,FX,FY,TX,TY, PX,PY,EX).

atk_esc_v(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PX = EX, PY != EY,
    not blk_v_esc(C,PT,FX,FY,TX,TY, PX,PY,EY).

atk_esc_d(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    EX > PX, EY > PY, EX-PX = EY-PY,
    not blk_d_esc(C,PT,FX,FY,TX,TY, PX,PY,EX,EY).

atk_esc_d(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PX > EX, EY > PY, PX-EX = EY-PY,
    not blk_d_esc(C,PT,FX,FY,TX,TY, PX,PY,EX,EY).

atk_esc_d(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    EX > PX, PY > EY, EX-PX = PY-EY,
    not blk_d_esc(C,PT,FX,FY,TX,TY, PX,PY,EX,EY).

atk_esc_d(C,PT,FX,FY,TX,TY, PX,PY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PX > EX, PY > EY, PX-EX = PY-EY,
    not blk_d_esc(C,PT,FX,FY,TX,TY, PX,PY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = queen,
    atk_esc_h(C,PT,FX,FY,TX,TY, TX,TY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = queen,
    atk_esc_v(C,PT,FX,FY,TX,TY, TX,TY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = queen,
    atk_esc_d(C,PT,FX,FY,TX,TY, TX,TY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = rook,
    atk_esc_h(C,PT,FX,FY,TX,TY, TX,TY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = rook,
    atk_esc_v(C,PT,FX,FY,TX,TY, TX,TY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = bishop,
    atk_esc_d(C,PT,FX,FY,TX,TY, TX,TY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = knight,
    knight_delta(DX,DY),
    EX = TX+DX, EY = TY+DY,
    file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = pawn, C = white,
    EY = TY+1, EX = TX+1, file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = pawn, C = white,
    EY = TY+1, TX = EX+1, file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = pawn, C = black,
    TY = EY+1, EX = TX+1, file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = pawn, C = black,
    TY = EY+1, TX = EX+1, file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT = king,
    king_delta(DX,DY),
    EX = TX+DX, EY = TY+DY,
    file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    mover_piece_after(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX != TX,
    atk_esc_h(C,PT,FX,FY,TX,TY, QX,QY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    mover_piece_after(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX != TX,
    atk_esc_v(C,PT,FX,FY,TX,TY, QX,QY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    mover_piece_after(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX != TX,
    atk_esc_d(C,PT,FX,FY,TX,TY, QX,QY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    mover_piece_after(C,PT,FX,FY,TX,TY, rook,RX,RY),
    RX != TX,
    atk_esc_h(C,PT,FX,FY,TX,TY, RX,RY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    mover_piece_after(C,PT,FX,FY,TX,TY, rook,RX,RY),
    RX != TX,
    atk_esc_v(C,PT,FX,FY,TX,TY, RX,RY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    mover_piece_after(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    BX != TX,
    atk_esc_d(C,PT,FX,FY,TX,TY, BX,BY,EX,EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    mover_piece_after(C,PT,FX,FY,TX,TY, knight,NX,NY),
    NX != TX,
    knight_delta(DX,DY),
    EX = NX+DX, EY = NY+DY,
    file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    C = white,
    mover_piece_after(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PX != TX,
    EY = PY+1, EX = PX+1, file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    C = white,
    mover_piece_after(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PX != TX,
    EY = PY+1, PX = EX+1, file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    C = black,
    mover_piece_after(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PX != TX,
    PY = EY+1, EX = PX+1, file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    C = black,
    mover_piece_after(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PX != TX,
    PY = EY+1, PX = EX+1, file(EX), rank(EY).

sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY), PT != king,
    mover_king_after(C,PT,FX,FY,TX,TY, KX,KY),
    king_delta(DX,DY),
    EX = KX+DX, EY = KY+DY,
    file(EX), rank(EY).

opp_own_piece_after(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,_,EX,EY),
    EX != TX.

opp_own_piece_after(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,_,EX,EY),
    EY != TY.

opp_own_piece_after(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,EX,EY).

king_can_escape(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    king_delta(DX,DY),
    EX = OKX+DX, EY = OKY+DY,
    file(EX), rank(EY),
    not opp_own_piece_after(C,PT,FX,FY,TX,TY, EX,EY),
    not sq_attacked_by_mover(C,PT,FX,FY,TX,TY, EX,EY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,queen,PX,PY),
    PY = TY, PX != TX,
    not blocked_h(PX,PY,TX),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,queen,PX,PY),
    PX = TX, PY != TY,
    not blocked_v(PX,PY,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,queen,PX,PY),
    TX > PX, TY > PY, TX-PX = TY-PY,
    not blocked_d(PX,PY,TX,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,queen,PX,PY),
    PX > TX, TY > PY, PX-TX = TY-PY,
    not blocked_d(PX,PY,TX,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,queen,PX,PY),
    TX > PX, PY > TY, TX-PX = PY-TY,
    not blocked_d(PX,PY,TX,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,queen,PX,PY),
    PX > TX, PY > TY, PX-TX = PY-TY,
    not blocked_d(PX,PY,TX,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,rook,PX,PY),
    PY = TY, PX != TX,
    not blocked_h(PX,PY,TX),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,rook,PX,PY),
    PX = TX, PY != TY,
    not blocked_v(PX,PY,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,bishop,PX,PY),
    TX > PX, TY > PY, TX-PX = TY-PY,
    not blocked_d(PX,PY,TX,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,bishop,PX,PY),
    PX > TX, TY > PY, PX-TX = TY-PY,
    not blocked_d(PX,PY,TX,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,bishop,PX,PY),
    TX > PX, PY > TY, TX-PX = PY-TY,
    not blocked_d(PX,PY,TX,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,bishop,PX,PY),
    PX > TX, PY > TY, PX-TX = PY-TY,
    not blocked_d(PX,PY,TX,TY),
    not own_piece(Opp,TX,TY).

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,knight,PX,PY),
    knight_delta(DX,DY),
    TX = PX+DX, TY = PY+DY.

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp), C = white,
    piece(Opp,pawn,PX,PY),
    TY = PY-1, TX = PX+1.

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp), C = white,
    piece(Opp,pawn,PX,PY),
    TY = PY-1, PX = TX+1.

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp), C = black,
    piece(Opp,pawn,PX,PY),
    TY = PY+1, TX = PX+1.

opp_can_capture_checker(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp), C = black,
    piece(Opp,pawn,PX,PY),
    TY = PY+1, PX = TX+1.

checker_defended(C,PT,FX,FY,TX,TY) :-
    sq_attacked_by_mover(C,PT,FX,FY,TX,TY, TX,TY).

opp_can_escape(C,PT,FX,FY,TX,TY) :-
    king_can_escape(C,PT,FX,FY,TX,TY, _,_).

opp_can_escape(C,PT,FX,FY,TX,TY) :-
    opp_can_capture_checker(C,PT,FX,FY,TX,TY),
    not checker_defended(C,PT,FX,FY,TX,TY).

is_checkmate(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    in_check_after(C,PT,FX,FY,TX,TY),
    not opp_can_escape(C,PT,FX,FY,TX,TY).

mate_move(C,PT,FX,FY,TX,TY) :-
    is_checkmate(C,PT,FX,FY,TX,TY).

#show mate_move/6.