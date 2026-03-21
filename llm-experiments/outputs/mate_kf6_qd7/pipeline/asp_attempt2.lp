Looking at the previous attempt's errors, the issue is with `not (AX = KX, AY = KY)` style negation of conjunctions in rule bodies, and similar constructs. I need to use helper predicates instead.

Let me rewrite the program carefully, avoiding all problematic syntax.

% Domain definitions
file(1..8).
rank(1..8).

% Helper: same square
same_sq(X1,Y1,X2,Y2) :- file(X1), rank(Y1), file(X2), rank(Y2), X1=X2, Y1=Y2.

% All occupied squares
occupied(X,Y) :- piece(_,_,X,Y).
occupied(X,Y) :- king(_,X,Y).

% Color helpers
white_king(KX,KY) :- king(white,KX,KY).
black_king(KX,KY) :- king(black,KX,KY).

% Own (white) pieces
own_piece(X,Y) :- piece(white,_,X,Y).
own_piece(X,Y) :- king(white,X,Y).

%-----------------------------------------------------------------------
% Queen move generation: horizontal, vertical, diagonal
%-----------------------------------------------------------------------
queen_move_h(FX,FY,TX,FY) :-
    piece(white,queen,FX,FY), file(TX), TX != FX.

queen_move_v(FX,FY,FX,TY) :-
    piece(white,queen,FX,FY), rank(TY), TY != FY.

queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX != FX,
    D = TX - FX, D > 0,
    D2 = TY - FY, D2 > 0,
    D = D2.

queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX != FX,
    D = FX - TX, D > 0,
    D2 = TY - FY, D2 > 0,
    D = D2.

queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX != FX,
    D = TX - FX, D > 0,
    D2 = FY - TY, D2 > 0,
    D = D2.

queen_move_d(FX,FY,TX,TY) :-
    piece(white,queen,FX,FY), file(TX), rank(TY),
    TX != FX,
    D = FX - TX, D > 0,
    D2 = FY - TY, D2 > 0,
    D = D2.

%-----------------------------------------------------------------------
% Path blocking
%-----------------------------------------------------------------------
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

%-----------------------------------------------------------------------
% Valid queen moves
%-----------------------------------------------------------------------
queen_can_move(FX,FY,TX,TY) :-
    queen_move_h(FX,FY,TX,TY),
    not blocked_h(FX,FY,TX),
    not own_piece(TX,TY).

queen_can_move(FX,FY,TX,TY) :-
    queen_move_v(FX,FY,TX,TY),
    not blocked_v(FX,FY,TY),
    not own_piece(TX,TY).

queen_can_move(FX,FY,TX,TY) :-
    queen_move_d(FX,FY,TX,TY),
    not blocked_d(FX,FY,TX,TY),
    not own_piece(TX,TY).

%-----------------------------------------------------------------------
% Occupied squares after queen moves from (FX,FY) to (QX,QY)
%-----------------------------------------------------------------------
not_from(BX,BY,FX,FY) :- occupied(BX,BY), BX != FX.
not_from(BX,BY,FX,FY) :- occupied(BX,BY), BY != FY.

not_to(BX,BY,QX,QY) :- occupied(BX,BY), BX != QX.
not_to(BX,BY,QX,QY) :- occupied(BX,BY), BY != QY.

occ_after(BX,BY,FX,FY,QX,QY) :-
    occupied(BX,BY),
    not_from(BX,BY,FX,FY),
    not_to(BX,BY,QX,QY).

occ_after(BX,BY,FX,FY,QX,QY) :-
    occupied(BX,BY),
    not_from(BX,BY,FX,FY),
    BX = QX, BY = QY.

occ_after(QX,QY,FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY).

%-----------------------------------------------------------------------
% Blocking after move (for queen now at QX,QY attacking AX,AY)
%-----------------------------------------------------------------------
blk_h_after(QX,QY,AX,FX,FY,QX,QY) :-
    occ_after(BX,QY,FX,FY,QX,QY),
    QX < AX, BX > QX, BX < AX.
blk_h_after(QX,QY,AX,FX,FY,QX,QY) :-
    occ_after(BX,QY,FX,FY,QX,QY),
    QX > AX, BX < QX, BX > AX.

blk_v_after(QX,QY,AY,FX,FY,QX,QY) :-
    occ_after(QX,BY,FX,FY,QX,QY),
    QY < AY, BY > QY, BY < AY.
blk_v_after(QX,QY,AY,FX,FY,QX,QY) :-
    occ_after(QX,BY,FX,FY,QX,QY),
    QY > AY, BY < QY, BY > AY.

blk_d_after(QX,QY,AX,AY,FX,FY,QX,QY) :-
    occ_after(BX,BY,FX,FY,QX,QY),
    QX < AX, QY < AY, BX > QX, BX < AX, BY > QY, BY < AY,
    BX - QX = BY - QY.

blk_d_after(QX,QY,AX,AY,FX,FY,QX,QY) :-
    occ_after(BX,BY,FX,FY,QX,QY),
    QX > AX, QY < AY, BX < QX, BX > AX, BY > QY, BY < AY,
    QX - BX = BY - QY.

blk_d_after(QX,QY,AX,AY,FX,FY,QX,QY) :-
    occ_after(BX,BY,FX,FY,QX,QY),
    QX < AX, QY > AY, BX > QX, BX < AX, BY < QY, BY > AY,
    BX - QX = QY - BY.

blk_d_after(QX,QY,AX,AY,FX,FY,QX,QY) :-
    occ_after(BX,BY,FX,FY,QX,QY),
    QX > AX, QY > AY, BX < QX, BX > AX, BY < QY, BY > AY,
    QX - BX = QY - BY.

%-----------------------------------------------------------------------
% Queen (at QX,QY) attacks (AX,AY) after moving from (FX,FY)
%-----------------------------------------------------------------------
q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AY = QY, AX != QX,
    not blk_h_after(QX,QY,AX,FX,FY,QX,QY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX = QX, AY != QY,
    not blk_v_after(QX,QY,AY,FX,FY,QX,QY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX != QX,
    D = AX - QX, D > 0, D2 = AY - QY, D2 > 0, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY,QX,QY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX != QX,
    D = QX - AX, D > 0, D2 = AY - QY, D2 > 0, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY,QX,QY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX != QX,
    D = AX - QX, D > 0, D2 = QY - AY, D2 > 0, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY,QX,QY).

q_att_after(QX,QY,AX,AY,FX,FY) :-
    queen_can_move(FX,FY,QX,QY),
    file(AX), rank(AY), AX != QX,
    D = QX - AX, D > 0, D2 = QY - AY, D2 > 0, D = D2,
    not blk_d_after(QX,QY,AX,AY,FX,FY,QX,QY).

%-----------------------------------------------------------------------
% White king attacks (king doesn't move)
%-----------------------------------------------------------------------
diff_le1_pos(D) :- D = 0.
diff_le1_pos(D) :- D = 1.

king_att(KX,KY,AX,AY) :-
    king(white,KX,KY), file(AX), rank(AY),
    DX = AX - KX, diff_le1_pos(DX),
    DY = AY - KY, diff_le1_pos(DY),
    AX != KX.

king_att(KX,KY,AX,AY) :-
    king(white,KX,KY), file(AX), rank(AY),
    DX = KX - AX, diff_le1_pos(DX),
    DY = AY - KY, diff_le1_pos(DY),
    AX != KX.

king_att(KX,KY,AX,AY) :-
    king(white,KX,KY), file(AX), rank(AY),
    DX = AX - KX, diff_le1_pos(DX),
    DY = KY - AY, diff_le1_pos(DY),
    AY != KY, AX = KX.

king_att(KX,KY,AX,AY) :-
    king(white,KX,KY), file(AX), rank(AY),
    DX = KX - AX, diff_le1_pos(DX),
    DY = KY - AY, diff_le1_pos(DY),
    AY != KY, AX = KX.

%-----------------------------------------------------------------------
% White attacks square after queen move
%-----------------------------------------------------------------------
white_att_after(AX,AY,FX,FY,QX,QY) :-
    q_att_after(QX,QY,AX,AY,FX,FY).

white_att_after(AX,AY,FX,FY,QX,QY) :-
    king_att(KX,KY,AX,AY),
    queen_can_move(FX,FY,QX,QY).

%-----------------------------------------------------------------------
% White pieces on board after move
%-----------------------------------------------------------------------
white_occ_after(X,Y,FX,FY,QX,QY) :-
    piece(white,_,X,Y), X != FX.
white_occ_after(X,Y,FX,FY,QX,QY) :-
    piece(white,_,X,Y), Y != FY.
white_occ_after(QX,QY,FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY).
white_occ_after(X,Y,FX,FY,QX,QY) :-
    king(white,X,Y), queen_can_move(FX,FY,QX,QY).

%-----------------------------------------------------------------------
% Adjacent squares to black king
%-----------------------------------------------------------------------
adj_bk(AX,AY) :-
    black_king(KX,KY), file(AX), rank(AY),
    DX = AX - KX, diff_le1_pos(DX),
    DY = AY - KY, diff_le1_pos(DY),
    AX != KX.

adj_bk(AX,AY) :-
    black_king(KX,KY), file(AX), rank(AY),
    DX = KX - AX, diff_le1_pos(DX),
    DY = AY - KY, diff_le1_pos(DY),
    AX != KX.

adj_bk(AX,AY) :-
    black_king(KX,KY), file(AX), rank(AY),
    DX = AX - KX, diff_le1_pos(DX),
    DY = KY - AY, diff_le1_pos(DY),
    AY != KY, AX = KX.

adj_bk(AX,AY) :-
    black_king(KX,KY), file(AX), rank(AY),
    DX = KX - AX, diff_le1_pos(DX),
    DY = KY - AY, diff_le1_pos(DY),
    AY != KY, AX = KX.

%-----------------------------------------------------------------------
% Black king in check after move
%-----------------------------------------------------------------------
in_check_after(FX,FY,QX,QY) :-
    black_king(KX,KY),
    white_att_after(KX,KY,FX,FY,QX,QY).

%-----------------------------------------------------------------------
% Queen defended after move (white king covers queen square)
%-----------------------------------------------------------------------
queen_defended_after(QX,QY) :-
    king_att(KX,KY,QX,QY).

%-----------------------------------------------------------------------
% Can black king escape to (AX,AY) after white queen moves (FX,FY)->(QX,QY)?
% Escape: adjacent, not attacked by white, not blocked by white piece
% (unless it's undefended queen)
%-----------------------------------------------------------------------

% Black king can't go to square attacked by white
% Black king can't go to square with white piece (unless captures undefended queen)
% Black has no other pieces

% Square is safe for black king to move to:
bk_can_go(AX,AY,FX,FY,QX,QY) :-
    adj_bk(AX,AY),
    not white_att_after(AX,AY,FX,FY,QX,QY),
    not white_occ_after(AX,AY,FX,FY,QX,QY).

% Black king captures queen: queen at (QX,QY), adjacent to black king, not defended
bk_can_go(QX,QY,FX,FY,QX,QY) :-
    adj_bk(QX,QY),
    queen_can_move(FX,FY,QX,QY),
    not queen_defended_after(QX,QY),
    not white_att_after(QX,QY,FX,FY,QX,QY).

%-----------------------------------------------------------------------
% Checkmate
%-----------------------------------------------------------------------
is_checkmate(FX,FY,QX,QY) :-
    queen_can_move(FX,FY,QX,QY),
    in_check_after(FX,FY,QX,QY),
    not bk_can_go(_,_,FX,FY,QX,QY).

%-----------------------------------------------------------------------
% Output
%-----------------------------------------------------------------------
mate_move(white,queen,FX,FY,QX,QY) :-
    is_checkmate(FX,FY,QX,QY).

#show mate_move/6.