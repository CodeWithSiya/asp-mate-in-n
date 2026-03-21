Looking at the previous attempt's errors, the main issues were:
1. `\=` used inside `not(...)` conjunctions (not valid in Clingo negation-as-failure)
2. Complex negation with conjunctions inside `not`

I need to rewrite using helper predicates to avoid these issues, and ensure all Clingo syntax is correct.

The key insight: Queen on h1 (8,1) moves to h8 (8,8) along file h. The pawn on h2 (8,2) blocks this. However, looking at the position carefully - the queen can move along rank 1, and the mating move based on the CoT is Queen (8,1) to (8,8). The ASP should find this correctly if the blocking logic is right.

Let me write a clean, correct ASP program:

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

blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), TX>FX, TY>FY, BX>FX, BX<TX, BY>FY, BY<TY, BX-FX=BY-FY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), TX<FX, TY>FY, BX<FX, BX>TX, BY>FY, BY<TY, FX-BX=BY-FY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), TX>FX, TY<FY, BX>FX, BX<TX, BY<FY, BY>TY, BX-FX=FY-BY.
blocked_d(FX,FY,TX,TY) :- occupied(BX,BY), TX<FX, TY<FY, BX<FX, BX>TX, BY<FY, BY>TY, FX-BX=FY-BY.

% ---- queen moves ----
queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FY=TY, FX != TX,
    not blocked_h(FX,FY,TX), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX=TX, FY != TY,
    not blocked_v(FX,FY,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), TX>FX, TY>FY, TX-FX=TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX>TX, TY>FY, FX-TX=TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), TX>FX, FY>TY, TX-FX=FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

queen_move(FX,FY,TX,TY) :-
    piece(C,queen,FX,FY), to_move(C),
    file(TX), rank(TY), FX>TX, FY>TY, FX-TX=FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

% ---- rook moves ----
rook_move(FX,FY,TX,TY) :-
    piece(C,rook,FX,FY), to_move(C),
    file(TX), rank(TY), FY=TY, FX != TX,
    not blocked_h(FX,FY,TX), not own_piece(C,TX,TY).

rook_move(FX,FY,TX,TY) :-
    piece(C,rook,FX,FY), to_move(C),
    file(TX), rank(TY), FX=TX, FY != TY,
    not blocked_v(FX,FY,TY), not own_piece(C,TX,TY).

% ---- bishop moves ----
bishop_move(FX,FY,TX,TY) :-
    piece(C,bishop,FX,FY), to_move(C),
    file(TX), rank(TY), TX>FX, TY>FY, TX-FX=TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

bishop_move(FX,FY,TX,TY) :-
    piece(C,bishop,FX,FY), to_move(C),
    file(TX), rank(TY), FX>TX, TY>FY, FX-TX=TY-FY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

bishop_move(FX,FY,TX,TY) :-
    piece(C,bishop,FX,FY), to_move(C),
    file(TX), rank(TY), TX>FX, FY>TY, TX-FX=FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

bishop_move(FX,FY,TX,TY) :-
    piece(C,bishop,FX,FY), to_move(C),
    file(TX), rank(TY), FX>TX, FY>TY, FX-TX=FY-TY,
    not blocked_d(FX,FY,TX,TY), not own_piece(C,TX,TY).

% ---- knight moves ----
knight_delta(2,1). knight_delta(2,-1). knight_delta(-2,1). knight_delta(-2,-1).
knight_delta(1,2). knight_delta(1,-2). knight_delta(-1,2). knight_delta(-1,-2).

knight_move(FX,FY,TX,TY) :-
    piece(C,knight,FX,FY), to_move(C),
    knight_delta(DX,DY),
    TX=FX+DX, TY=FY+DY,
    file(TX), rank(TY),
    not own_piece(C,TX,TY).

% ---- pawn moves ----
pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    TX=FX, TY=FY+1, file(TX), rank(TY),
    not occupied(TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    FY=2, TX=FX, TY=4, file(TX), rank(TY),
    not occupied(FX,3), not occupied(FX,4).

pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    TY=FY+1, TX=FX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(white,TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(white,pawn,FX,FY), to_move(white),
    TY=FY+1, FX=TX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(white,TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    TX=FX, TY=FY-1, file(TX), rank(TY),
    not occupied(TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    FY=7, TX=FX, TY=5, file(TX), rank(TY),
    not occupied(FX,6), not occupied(FX,5).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    TY=FY-1, TX=FX+1, file(TX), rank(TY),
    occupied(TX,TY), not own_piece(black,TX,TY).

pawn_move(FX,FY,TX,TY) :-
    piece(black,pawn,FX,FY), to_move(black),
    TY=FY-1, FX=TX+1, file(TX), rank(TY),
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

% ---- board state after move ----
% A square is occupied after move M=(C,PT,FX,FY,TX,TY) if:
%   - it had a piece that wasn't moved and wasn't captured
%   - or it's the destination of the moved piece

% piece still there after move
piece_stays(C,PT,FX,FY,TX,TY, PC,PPT,PX,PY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece(PC,PPT,PX,PY),
    not moved_piece(C,PT,FX,FY,TX,TY,PC,PPT,PX,PY),
    not captured_at(TX,TY,PX,PY).

moved_piece(C,PT,FX,FY,TX,TY,C,PT,FX,FY) :- cand(C,PT,FX,FY,TX,TY).
captured_at(TX,TY,TX,TY) :- file(TX), rank(TY).

% occupied after move
occ_after(C,PT,FX,FY,TX,TY, SX,SY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece_stays(C,PT,FX,FY,TX,TY, _,_,SX,SY).

occ_after(C,PT,FX,FY,TX,TY, TX,TY) :-
    cand(C,PT,FX,FY,TX,TY).

% kings stay (unless king moved)
occ_after(C,PT,FX,FY,TX,TY, KX,KY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(KC,KX,KY),
    PT = king, KC = C,
    KX = TX, KY = TY.  % king moved to TX,TY - already covered above

occ_after(C,PT,FX,FY,TX,TY, KX,KY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(KC,KX,KY),
    PT != king,
    not (KX = TX, KY = TY).  % not captured

occ_after(C,PT,FX,FY,TX,TY, KX,KY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(KC,KX,KY),
    PT = king, KC != C.

% own_after: squares occupied by Color after move
own_after(C,PT,FX,FY,TX,TY, C, SX,SY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece_stays(C,PT,FX,FY,TX,TY, C,_,SX,SY).

own_after(C,PT,FX,FY,TX,TY, C, TX,TY) :-
    cand(C,PT,FX,FY,TX,TY).

own_after(C,PT,FX,FY,TX,TY, C, KX,KY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(C,KX,KY),
    PT != king.

own_after(C,PT,FX,FY,TX,TY, C, TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = king.

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
    TF>SF, TR>SR, BX>SF, BX<TF, BY>SR, BY<TR, BX-SF=BY-SR.
blk_d_after(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,BY),
    SF>TF, TR>SR, BX<SF, BX>TF, BY>SR, BY<TR, SF-BX=BY-SR.
blk_d_after(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,BY),
    TF>SF, SR>TR, BX>SF, BX<TF, BY<SR, BY>TR, BX-SF=SR-BY.
blk_d_after(C,PT,FX,FY,TX,TY, SF,SR,TF,TR) :-
    cand(C,PT,FX,FY,TX,TY),
    occ_after(C,PT,FX,FY,TX,TY, BX,BY),
    SF>TF, SR>TR, BX<SF, BX>TF, BY<SR, BY>TR, SF-BX=SR-BY.

% ---- attack detection after move ----
% Does Color's piece at (PX,PY) attack (AX,AY) after move, given board occ_after?

% Queen at (QX,QY) attacks (AX,AY)
q_atk_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    QY=AY, QX != AX,
    not blk_h_after(C,PT,FX,FY,TX,TY, QX,QY,AX).

q_atk_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    QX=AX, QY != AY,
    not blk_v_after(C,PT,FX,FY,TX,TY, QX,QY,AY).

q_atk_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    AX>QX, AY>QY, AX-QX=AY-QY,
    not blk_d_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY).

q_atk_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    QX>AX, AY>QY, QX-AX=AY-QY,
    not blk_d_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY).

q_atk_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    AX>QX, QY>AY, AX-QX=QY-AY,
    not blk_d_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY).

q_atk_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    QX>AX, QY>AY, QX-AX=QY-AY,
    not blk_d_after(C,PT,FX,FY,TX,TY, QX,QY,AX,AY).

% Rook at (RX,RY) attacks (AX,AY)
r_atk_after(C,PT,FX,FY,TX,TY, RX,RY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    RY=AY, RX != AX,
    not blk_h_after(C,PT,FX,FY,TX,TY, RX,RY,AX).

r_atk_after(C,PT,FX,FY,TX,TY, RX,RY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    RX=AX, RY != AY,
    not blk_v_after(C,PT,FX,FY,TX,TY, RX,RY,AY).

% Bishop at (BX,BY) attacks (AX,AY)
b_atk_after(C,PT,FX,FY,TX,TY, BPX,BPY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    AX>BPX, AY>BPY, AX-BPX=AY-BPY,
    not blk_d_after(C,PT,FX,FY,TX,TY, BPX,BPY,AX,AY).

b_atk_after(C,PT,FX,FY,TX,TY, BPX,BPY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    BPX>AX, AY>BPY, BPX-AX=AY-BPY,
    not blk_d_after(C,PT,FX,FY,TX,TY, BPX,BPY,AX,AY).

b_atk_after(C,PT,FX,FY,TX,TY, BPX,BPY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    AX>BPX, BPY>AY, AX-BPX=BPY-AY,
    not blk_d_after(C,PT,FX,FY,TX,TY, BPX,BPY,AX,AY).

b_atk_after(C,PT,FX,FY,TX,TY, BPX,BPY,AX,AY) :-
    cand(C,PT,FX,FY,TX,TY),
    BPX>AX, BPY>AY, BPX-AX=BPY-AY,
    not blk_d_after(C,PT,FX,FY,TX,TY, BPX,BPY,AX,AY).

% ---- attacks on opponent king after move ----
% We collect all Color pieces after move and check if any attacks opp king

% Moved piece position and type
moved_to_type(C,PT,FX,FY,TX,TY, PT,TX,TY) :- cand(C,PT,FX,FY,TX,TY).

% Remaining pieces of Color (not moved, not captured)
rem_piece(C,PT,FX,FY,TX,TY, PPT,PX,PY) :-
    cand(C,PT,FX,FY,TX,TY),
    piece(C,PPT,PX,PY),
    not (PX=FX, PY=FY, PPT=PT),
    not (PX=TX, PY=TY).

% Opponent king position (not captured)
opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    not (TX=OKX, TY=OKY).

% Is opponent king in check after move?
in_check_after(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp), king(Opp,OKX,OKY),
    TX=OKX, TY=OKY.  % captured king

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT=queen,
    q_atk_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT=rook,
    r_atk_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT=bishop,
    b_atk_after(C,PT,FX,FY,TX,TY, TX,TY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT=knight,
    knight_delta(DX,DY),
    OKX=TX+DX, OKY=TY+DY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    C=white, PT=pawn,
    OKY=TY+1, OKX=TX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    C=white, PT=pawn,
    OKY=TY+1, TX=OKX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    C=black, PT=pawn,
    TY=OKY+1, OKX=TX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    C=black, PT=pawn,
    TY=OKY+1, TX=OKX+1.

% Check from remaining pieces
in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    q_atk_after(C,PT,FX,FY,TX,TY, QX,QY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, rook,RX,RY),
    r_atk_after(C,PT,FX,FY,TX,TY, RX,RY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    b_atk_after(C,PT,FX,FY,TX,TY, BX,BY,OKX,OKY).

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    rem_piece(C,PT,FX,FY,TX,TY, knight,NX,NY),
    knight_delta(DX,DY),
    OKX=NX+DX, OKY=NY+DY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    C=white,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    OKY=PY+1, OKX=PX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    C=white,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    OKY=PY+1, PX=OKX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    C=black,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PY=OKY+1, OKX=PX+1.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    C=black,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PY=OKY+1, PX=OKX+1.

% King attacks
in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    king(C,KX,KY),
    PT != king,
    OKX-KX <= 1, KX-OKX <= 1,
    OKY-KY <= 1, KY-OKY <= 1,
    OKX != KX.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    king(C,KX,KY),
    PT != king,
    OKX-KX <= 1, KX-OKX <= 1,
    OKY-KY <= 1, KY-OKY <= 1,
    OKY != KY.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = king,
    OKX-TX <= 1, TX-OKX <= 1,
    OKY-TY <= 1, TY-OKY <= 1,
    OKX != TX.

in_check_after(C,PT,FX,FY,TX,TY) :-
    opp_king_sq(C,PT,FX,FY,TX,TY, OKX,OKY),
    PT = king,
    OKX-TX <= 1, TX-OKX <= 1,
    OKY-TY <= 1, TY-OKY <= 1,
    OKY != TY.

% ============================================================
% Opponent escape: can opponent king move to safety?
% After move (C,PT,FX,FY,TX,TY), opponent king at (OKX,OKY) tries (EX,EY)
% ============================================================

% Board after move + opponent king moved from OKX,OKY to EX,EY
% Occupied squares: occ_after minus OKX,OKY, plus EX,EY (king)

% Helper: is sq (SX,SY) occupied after move AND after opp king moves to (EX,EY)?
occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SX,SY) :-
    occ_after(C,PT,FX,FY,TX,TY, SX,SY),
    not (SX=OKX, SY=OKY).

% Blocking after king escape
blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,SR,TR) :-
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,SR),
    SF < TR, BX > SF, BX < TR.
blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,SR,TR) :-
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,SR),
    SF > TR, BX < SF, BX > TR.

blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,SR,TR) :-
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,BY),
    SR < TR, BY > SR, BY < TR.
blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,SR,TR) :-
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,BY),
    SR > TR, BY < SR, BY > TR.

blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,SR,TF,TR) :-
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,BY),
    TF>SF, TR>SR, BX>SF, BX<TF, BY>SR, BY<TR, BX-SF=BY-SR.
blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,SR,TF,TR) :-
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,BY),
    SF>TF, TR>SR, BX<SF, BX>TF, BY>SR, BY<TR, SF-BX=BY-SR.
blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,SR,TF,TR) :-
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,BY),
    TF>SF, SR>TR, BX>SF, BX<TF, BY<SR, BY>TR, BX-SF=SR-BY.
blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, SF,SR,TF,TR) :-
    occ_after2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,BY),
    SF>TF, SR>TR, BX<SF, BX>TF, BY<SR, BY>TR, SF-BX=SR-BY.

% Attack on escape square EX,EY by Color's moved piece (at TX,TY)
atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=queen, EY=TY, EX != TX,
    not blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=queen, EX=TX, EY != TY,
    not blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=queen, EX>TX, EY>TY, EX-TX=EY-TY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=queen, TX>EX, EY>TY, TX-EX=EY-TY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=queen, EX>TX, TY>EY, EX-TX=TY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=queen, TX>EX, TY>EY, TX-EX=TY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=rook, EY=TY, EX != TX,
    not blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=rook, EX=TX, EY != TY,
    not blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=bishop, EX>TX, EY>TY, EX-TX=EY-TY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=bishop, TX>EX, EY>TY, TX-EX=EY-TY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=bishop, EX>TX, TY>EY, EX-TX=TY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=bishop, TX>EX, TY>EY, TX-EX=TY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, TX,TY,EX,EY).

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=knight, knight_delta(DX,DY), EX=TX+DX, EY=TY+DY.

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=pawn, C=white, EY=TY+1, EX=TX+1.

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=pawn, C=white, EY=TY+1, TX=EX+1.

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=pawn, C=black, TY=EY+1, EX=TX+1.

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=pawn, C=black, TY=EY+1, TX=EX+1.

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=king,
    EX-TX <= 1, TX-EX <= 1,
    EY-TY <= 1, TY-EY <= 1,
    EX != TX.

atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT=king,
    EX-TX <= 1, TX-EX <= 1,
    EY-TY <= 1, TY-EY <= 1,
    EY != TY.

% Attack on escape square by remaining Color pieces
atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    EY=QY, EX != QX,
    not blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, QX,QY,EX).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    EX=QX, EY != QY,
    not blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, QX,QY,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    EX>QX, EY>QY, EX-QX=EY-QY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, QX,QY,EX,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX>EX, EY>QY, QX-EX=EY-QY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, QX,QY,EX,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    EX>QX, QY>EY, EX-QX=QY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, QX,QY,EX,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, queen,QX,QY),
    QX>EX, QY>EY, QX-EX=QY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, QX,QY,EX,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, rook,RX,RY),
    EY=RY, EX != RX,
    not blk_h_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, RX,RY,EX).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, rook,RX,RY),
    EX=RX, EY != RY,
    not blk_v_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, RX,RY,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    EX>BX, EY>BY, EX-BX=EY-BY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,BY,EX,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    BX>EX, EY>BY, BX-EX=EY-BY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,BY,EX,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    EX>BX, BY>EY, EX-BX=BY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,BY,EX,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, bishop,BX,BY),
    BX>EX, BY>EY, BX-EX=BY-EY,
    not blk_d_a2(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY, BX,BY,EX,EY).

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    rem_piece(C,PT,FX,FY,TX,TY, knight,NX,NY),
    knight_delta(DX,DY), EX=NX+DX, EY=NY+DY.

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    C=white,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    EY=PY+1, EX=PX+1.

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    C=white,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    EY=PY+1, PX=EX+1.

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    C=black,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PY=EY+1, EX=PX+1.

atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    C=black,
    rem_piece(C,PT,FX,FY,TX,TY, pawn,PX,PY),
    PY=EY+1, PX=EX+1.

% King attacks on escape square
atk_escape_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(C,KX,KY),
    PT != king,
    EX-KX <= 1, KX-EX <= 1,
    EY-KY <= 1, KY-EY <= 1,
    EX != KX.

atk_escape_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    king(C,KX,KY),
    PT != king,
    EX-KX <= 1, KX-EX <= 1,
    EY-KY <= 1, KY-EY <= 1,
    EY != KY.

atk_escape_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = king,
    EX-TX <= 1, TX-EX <= 1,
    EY-TY <= 1, TY-EY <= 1,
    EX != TX.

atk_escape_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    PT = king,
    EX-TX <= 1, TX-EX <= 1,
    EY-TY <= 1, TY-EY <= 1,
    EY != TY.

% Combined: escape square attacked
escape_sq_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    atk_escape_moved(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).
escape_sq_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    atk_escape_rem(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).
escape_sq_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY) :-
    atk_escape_king(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).

% Opponent color own pieces (for blocking king escape)
opp_own(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    piece(Opp,_,EX,EY),
    not (EX=TX, EY=TY).  % not captured

opp_own(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,EX,EY).

% Can opponent king escape to (EX,EY)?
% Must be adjacent, not own piece, not attacked, within board
king_can_escape(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    file(EX), rank(EY),
    EX-OKX <= 1, OKX-EX <= 1,
    EY-OKY <= 1, OKY-EY <= 1,
    EX != OKX,
    not opp_own(C,PT,FX,FY,TX,TY, EX,EY),
    not escape_sq_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).

king_can_escape(C,PT,FX,FY,TX,TY, EX,EY) :-
    cand(C,PT,FX,FY,TX,TY),
    opponent(C,Opp),
    king(Opp,OKX,OKY),
    file(EX), rank(EY),
    EX-OKX <= 1, OKX-EX <= 1,
    EY-OKY <= 1, OKY-EY <= 1,
    EY != OKY,
    not opp_own(C,PT,FX,FY,TX,TY, EX,EY),
    not escape_sq_attacked(C,PT,FX,FY,TX,TY,OKX,OKY,EX,EY).

% Checkmate
is_checkmate(C,PT,FX,FY,TX,TY) :-
    cand(C,PT,FX,FY,TX,TY),
    in_check_after(C,PT,FX,FY,TX,TY),
    not king_can_escape(C,PT,FX,FY,TX,TY,_,_).

mate_move(C,PT,FX,FY,TX,TY) :-
    is_checkmate(C,PT,FX,FY,TX,TY).

#show mate_move/6.