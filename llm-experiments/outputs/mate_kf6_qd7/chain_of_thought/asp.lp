% Domain definitions
file(1..8).
rank(1..8).

% Occupied squares (by any piece)
occupied(X, Y) :- piece(_, _, X, Y).
occupied(X, Y) :- king(_, X, Y).

% Occupied by white
occupied_white(X, Y) :- piece(white, _, X, Y).
occupied_white(X, Y) :- king(white, X, Y).

% Occupied by black
occupied_black(X, Y) :- piece(black, _, X, Y).
occupied_black(X, Y) :- king(black, X, Y).

% -------------------------------------------------------
% Blocking helpers for sliding pieces
% -------------------------------------------------------

% Horizontal blocking: something between (FX,FY) and (TX,FY) on same rank
blocked_h(FX, FY, TX) :-
    piece(_, _, BX, FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    king(_, BX, FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    piece(_, _, BX, FY), FX > TX, BX < FX, BX > TX.
blocked_h(FX, FY, TX) :-
    king(_, BX, FY), FX > TX, BX < FX, BX > TX.

% Vertical blocking
blocked_v(FX, FY, TY) :-
    piece(_, _, FX, BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    king(_, FX, BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    piece(_, _, FX, BY), FY > TY, BY < FY, BY > TY.
blocked_v(FX, FY, TY) :-
    king(_, FX, BY), FY > TY, BY < FY, BY > TY.

% Diagonal blocking: need to check squares strictly between on a diagonal
% Helper: is (BX,BY) strictly between (FX,FY) and (TX,TY) on a diagonal?
on_diag_between(FX, FY, TX, TY, BX, BY) :-
    file(BX), rank(BY),
    TX - FX = TY - FY,  % same diagonal direction (+,+) or (-,-)
    TX - FX > 0,
    BX > FX, BX < TX,
    BY - FY = BX - FX.
on_diag_between(FX, FY, TX, TY, BX, BY) :-
    file(BX), rank(BY),
    TX - FX = TY - FY,  % same diagonal direction
    TX - FX < 0,
    BX < FX, BX > TX,
    BY - FY = BX - FX.
on_diag_between(FX, FY, TX, TY, BX, BY) :-
    file(BX), rank(BY),
    FX - TX = TY - FY,  % anti-diagonal (+,-) or (-,+)
    TX - FX > 0,
    BX > FX, BX < TX,
    FY - BY = BX - FX.
on_diag_between(FX, FY, TX, TY, BX, BY) :-
    file(BX), rank(BY),
    FX - TX = TY - FY,  % anti-diagonal
    TX - FX < 0,
    BX < FX, BX > TX,
    FY - BY = BX - FX.

blocked_d(FX, FY, TX, TY) :-
    on_diag_between(FX, FY, TX, TY, BX, BY),
    piece(_, _, BX, BY).
blocked_d(FX, FY, TX, TY) :-
    on_diag_between(FX, FY, TX, TY, BX, BY),
    king(_, BX, BY).

% -------------------------------------------------------
% Queen move validity (white queen, not capturing own piece)
% -------------------------------------------------------
queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    FX = TX, FY != TY,
    not occupied_white(TX, TY),
    not blocked_v(FX, FY, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    FY = TY, FX != TX,
    not occupied_white(TX, TY),
    not blocked_h(FX, FY, TX).

queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX - FX = TY - FY,
    TX != FX,
    not occupied_white(TX, TY),
    not blocked_d(FX, FY, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    FX - TX = TY - FY,
    TX != FX,
    not occupied_white(TX, TY),
    not blocked_d(FX, FY, TX, TY).

% -------------------------------------------------------
% Squares attacked by white after queen moves to (TX,TY)
% (queen now at TX,TY; original square FX,FY vacated)
% -------------------------------------------------------

% Attacked by white king
attacked_by_white_king(SX, SY) :-
    king(white, KX, KY),
    file(SX), rank(SY),
    SX - KX <= 1, KX - SX <= 1,
    SY - KY <= 1, KY - SY <= 1,
    SX != KX.
attacked_by_white_king(SX, SY) :-
    king(white, KX, KY),
    file(SX), rank(SY),
    SX - KX <= 1, KX - SX <= 1,
    SY - KY <= 1, KY - SY <= 1,
    SY != KY.

% After queen moves from (FX,FY) to (QX,QY), is square (SX,SY) attacked by the queen?
% Horizontal
queen_attacks_after(FX, FY, QX, QY, SX, SY) :-
    file(SX), rank(SY),
    QX = SX, QY = SY, QX = QX.  % queen itself - not needed, skip

% Queen at (QX,QY) attacks (SX,SY) horizontally
queen_attacks_after(FX, FY, QX, QY, SX, SY) :-
    file(SX), rank(SY),
    QY = SY, QX != SX,
    not blocked_h_after(FX, FY, QX, QY, SX).

% Queen at (QX,QY) attacks (SX,SY) vertically
queen_attacks_after(FX, FY, QX, QY, SX, SY) :-
    file(SX), rank(SY),
    QX = SX, QY != SY,
    not blocked_v_after(FX, FY, QX, QY, SY).

% Queen at (QX,QY) attacks (SX,SY) diagonally
queen_attacks_after(FX, FY, QX, QY, SX, SY) :-
    file(SX), rank(SY),
    SX - QX = SY - QY, SX != QX,
    not blocked_d_after(FX, FY, QX, QY, SX, SY).

queen_attacks_after(FX, FY, QX, QY, SX, SY) :-
    file(SX), rank(SY),
    QX - SX = SY - QY, SX != QX,
    not blocked_d_after(FX, FY, QX, QY, SX, SY).

% Blocking after move: occupied squares change (FX,FY) is now empty, (QX,QY) is occupied by queen
% For horizontal blocking after move
blocked_h_after(FX, FY, QX, QY, SX) :-
    QY = QY,  % same rank as queen
    file(BX),
    QX < SX, BX > QX, BX < SX,
    occupied(BX, QY),
    BX != FX.  % original square is now empty
blocked_h_after(FX, FY, QX, QY, SX) :-
    QY = QY,
    file(BX),
    QX > SX, BX < QX, BX > SX,
    occupied(BX, QY),
    BX != FX.

% For vertical blocking after move
blocked_v_after(FX, FY, QX, QY, SY) :-
    QX = QX,
    rank(BY),
    QY < SY, BY > QY, BY < SY,
    occupied(QX, BY),
    BY != FY.
blocked_v_after(FX, FY, QX, QY, SY) :-
    QX = QX,
    rank(BY),
    QY > SY, BY < QY, BY > SY,
    occupied(QX, BY),
    BY != FY.

% For diagonal blocking after move
on_diag_between_after(FX, FY, QX, QY, SX, SY, BX, BY) :-
    file(BX), rank(BY),
    SX - QX = SY - QY,
    SX - QX > 0,
    BX > QX, BX < SX,
    BY - QY = BX - QX,
    BX != FX.
on_diag_between_after(FX, FY, QX, QY, SX, SY, BX, BY) :-
    file(BX), rank(BY),
    SX - QX = SY - QY,
    SX - QX < 0,
    BX < QX, BX > SX,
    BY - QY = BX - QX,
    BX != FX.
on_diag_between_after(FX, FY, QX, QY, SX, SY, BX, BY) :-
    file(BX), rank(BY),
    QX - SX = SY - QY,
    SX - QX > 0,
    BX > QX, BX < SX,
    QY - BY = BX - QX,
    BX != FX.
on_diag_between_after(FX, FY, QX, QY, SX, SY, BX, BY) :-
    file(BX), rank(BY),
    QX - SX = SY - QY,
    SX - QX < 0,
    BX < QX, BX > SX,
    QY - BY = BX - QX,
    BX != FX.

blocked_d_after(FX, FY, QX, QY, SX, SY) :-
    on_diag_between_after(FX, FY, QX, QY, SX, SY, BX, BY),
    occupied(BX, BY).

% -------------------------------------------------------
% Is the black king in check after white queen moves from (FX,FY) to (QX,QY)?
% -------------------------------------------------------
in_check_after(FX, FY, QX, QY) :-
    king(black, KX, KY),
    queen_attacks_after(FX, FY, QX, QY, KX, KY).

in_check_after(FX, FY, QX, QY) :-
    king(black, KX, KY),
    attacked_by_white_king(KX, KY).

% -------------------------------------------------------
% Can black king move to (TX,TY) after white queen is at (QX,QY)?
% -------------------------------------------------------
% Square (TX,TY) is attacked by white queen at (QX,QY) (queen came from FX,FY)
% Square (TX,TY) is attacked by white king
% Square (TX,TY) is occupied by black piece (can't move there unless capturing white - but white queen is at QX,QY)

% Black king adjacent squares
bk_adjacent(KX, KY, TX, TY) :-
    king(black, KX, KY),
    file(TX), rank(TY),
    TX - KX <= 1, KX - TX <= 1,
    TY - KY <= 1, KY - TY <= 1,
    TX != KX.
bk_adjacent(KX, KY, TX, TY) :-
    king(black, KX, KY),
    file(TX), rank(TY),
    TX - KX <= 1, KX - TX <= 1,
    TY - KY <= 1, KY - TY <= 1,
    TY != KY.

% Black king can escape to (TX,TY) if:
% - not occupied by black piece
% - not attacked by white queen after move
% - not attacked by white king
% - not the queen's square (occupied by white queen)
bk_can_escape(FX, FY, QX, QY, TX, TY) :-
    king(black, KX, KY),
    bk_adjacent(KX, KY, TX, TY),
    not occupied_black(TX, TY),
    not queen_attacks_after(FX, FY, QX, QY, TX, TY),
    not attacked_by_white_king(TX, TY),
    TX != QX.
bk_can_escape(FX, FY, QX, QY, TX, TY) :-
    king(black, KX, KY),
    bk_adjacent(KX, KY, TX, TY),
    not occupied_black(TX, TY),
    not queen_attacks_after(FX, FY, QX, QY, TX, TY),
    not attacked_by_white_king(TX, TY),
    TY != QY.

% Black king is stalemated or checkmated (no escape)
bk_no_escape(FX, FY, QX, QY) :-
    not bk_can_escape(FX, FY, QX, QY, _, _).

% -------------------------------------------------------
% Checkmate: in check and no escape
% -------------------------------------------------------
is_checkmate(FX, FY, QX, QY) :-
    in_check_after(FX, FY, QX, QY),
    bk_no_escape(FX, FY, QX, QY).

% -------------------------------------------------------
% Mate move
% -------------------------------------------------------
mate_move(white, queen, FX, FY, QX, QY) :-
    to_move(white),
    piece(white, queen, FX, FY),
    queen_can_move(FX, FY, QX, QY),
    is_checkmate(FX, FY, QX, QY).

#show mate_move/6.