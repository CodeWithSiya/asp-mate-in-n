% Domain
file(1..8).
rank(1..8).

% Opponent color
opponent(white, black).
opponent(black, white).

% All occupied squares
occupied(X, Y) :- piece(_, _, X, Y).
occupied(X, Y) :- king(_, X, Y).

% Occupied by a specific color
occupied_by(Color, X, Y) :- piece(Color, _, X, Y).
occupied_by(Color, X, Y) :- king(Color, X, Y).

% -------------------------------------------------------
% Queen move generation (8 directions)
% -------------------------------------------------------

queen_move(FX, FY, TX, FY) :-
    piece(white, queen, FX, FY),
    file(TX), TX > FX.

queen_move(FX, FY, TX, FY) :-
    piece(white, queen, FX, FY),
    file(TX), TX < FX.

queen_move(FX, FY, FX, TY) :-
    piece(white, queen, FX, FY),
    rank(TY), TY > FY.

queen_move(FX, FY, FX, TY) :-
    piece(white, queen, FX, FY),
    rank(TY), TY < FY.

queen_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX > FX, TY > FY,
    TX - FX = TY - FY.

queen_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX > FX, TY < FY,
    TX - FX = FY - TY.

queen_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX < FX, TY > FY,
    FX - TX = TY - FY.

queen_move(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX < FX, TY < FY,
    FX - TX = FY - TY.

% -------------------------------------------------------
% Blocking for queen moves
% -------------------------------------------------------

queen_blocked(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    occupied(BX, BY),
    TX > FX, TY = FY, BY = FY,
    BX > FX, BX < TX.

queen_blocked(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    occupied(BX, BY),
    TX < FX, TY = FY, BY = FY,
    BX < FX, BX > TX.

queen_blocked(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    occupied(BX, BY),
    TX = FX, TY > FY, BX = FX,
    BY > FY, BY < TY.

queen_blocked(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    occupied(BX, BY),
    TX = FX, TY < FY, BX = FX,
    BY < FY, BY > TY.

queen_blocked(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    occupied(BX, BY),
    TX > FX, TY > FY,
    TX - FX = TY - FY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.

queen_blocked(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    occupied(BX, BY),
    TX > FX, TY < FY,
    TX - FX = FY - TY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.

queen_blocked(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    occupied(BX, BY),
    TX < FX, TY > FY,
    FX - TX = TY - FY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.

queen_blocked(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    occupied(BX, BY),
    TX < FX, TY < FY,
    FX - TX = FY - TY,
    BX < FX, BX > TX,
    BY < FY, BY > TY,
    FX - BX = FY - BY.

% -------------------------------------------------------
% Valid queen moves
% -------------------------------------------------------

queen_can_move(FX, FY, TX, TY) :-
    queen_move(FX, FY, TX, TY),
    not queen_blocked(FX, FY, TX, TY),
    not occupied_by(white, TX, TY).

% -------------------------------------------------------
% Helper: same square
% -------------------------------------------------------

same_sq(X, Y, X, Y) :- file(X), rank(Y).

% -------------------------------------------------------
% Occupied squares after queen move from (FX,FY) to (TX,TY)
% -------------------------------------------------------

occ_after(TX, TY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY).

occ_after(X, Y, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    occupied(X, Y),
    not same_sq(X, Y, FX, FY).

% White occupied after move
occ_after_white(TX, TY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY).

occ_after_white(X, Y, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    occupied_by(white, X, Y),
    not same_sq(X, Y, FX, FY).

% -------------------------------------------------------
% White king attacks (static)
% -------------------------------------------------------

wking_adj(SX, SY) :-
    king(white, KX, KY),
    file(SX), rank(SY),
    SX >= KX - 1, SX =< KX + 1,
    SY >= KY - 1, SY =< KY + 1,
    not same_sq(SX, SY, KX, KY).

% -------------------------------------------------------
% Queen attacks from new position after move
% -------------------------------------------------------

% Horizontal right
queen_att_blocked_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY = TY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BY = TY, BX > TX, BX < SX.

queen_attacks_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY = TY,
    not queen_att_blocked_after(TX, TY, SX, SY, FX, FY).

% Horizontal left
queen_att_blocked_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY = TY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BY = TY, BX < TX, BX > SX.

queen_attacks_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY = TY,
    not queen_att_blocked_after(TX, TY, SX, SY, FX, FY).

% Vertical up
queen_att_blocked_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX = TX, SY > TY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX = TX, BY > TY, BY < SY.

queen_attacks_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX = TX, SY > TY,
    not queen_att_blocked_after(TX, TY, SX, SY, FX, FY).

% Vertical down
queen_att_blocked_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX = TX, SY < TY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX = TX, BY < TY, BY > SY.

queen_attacks_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX = TX, SY < TY,
    not queen_att_blocked_after(TX, TY, SX, SY, FX, FY).

% Diagonal NE
queen_att_blocked_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY > TY,
    SX - TX = SY - TY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX > TX, BX < SX,
    BY > TY, BY < SY,
    BX - TX = BY - TY.

queen_attacks_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY > TY,
    SX - TX = SY - TY,
    not queen_att_blocked_after(TX, TY, SX, SY, FX, FY).

% Diagonal SE
queen_att_blocked_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY < TY,
    SX - TX = TY - SY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX > TX, BX < SX,
    BY < TY, BY > SY,
    BX - TX = TY - BY.

queen_attacks_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY < TY,
    SX - TX = TY - SY,
    not queen_att_blocked_after(TX, TY, SX, SY, FX, FY).

% Diagonal NW
queen_att_blocked_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY > TY,
    TX - SX = SY - TY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX < TX, BX > SX,
    BY > TY, BY < SY,
    TX - BX = BY - TY.

queen_attacks_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY > TY,
    TX - SX = SY - TY,
    not queen_att_blocked_after(TX, TY, SX, SY, FX, FY).

% Diagonal SW
queen_att_blocked_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY < TY,
    TX - SX = TY - SY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX < TX, BX > SX,
    BY < TY, BY > SY,
    TX - BX = TY - BY.

queen_attacks_after(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY < TY,
    TX - SX = TY - SY,
    not queen_att_blocked_after(TX, TY, SX, SY, FX, FY).

% -------------------------------------------------------
% Combined white attacks after queen move
% -------------------------------------------------------

white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_attacks_after(TX, TY, SX, SY, FX, FY).

white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    wking_adj(SX, SY).

% -------------------------------------------------------
% Black king in check after white queen moves
% -------------------------------------------------------

black_king_in_check(FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    king(black, KX, KY),
    white_attacks_after(KX, KY, FX, FY, TX, TY).

% -------------------------------------------------------
% Black king escape squares
% -------------------------------------------------------

bking_escape_candidate(KX, KY, EX, EY) :-
    king(black, KX, KY),
    file(EX), rank(EY),
    EX >= KX - 1, EX =< KX + 1,
    EY >= KY - 1, EY =< KY + 1,
    not same_sq(EX, EY, KX, KY).

bking_can_escape(FX, FY, TX, TY, EX, EY) :-
    queen_can_move(FX, FY, TX, TY),
    king(black, KX, KY),
    bking_escape_candidate(KX, KY, EX, EY),
    not occupied_by(black, EX, EY),
    not white_attacks_after(EX, EY, FX, FY, TX, TY),
    not occ_after_white(EX, EY, FX, FY, TX, TY).

% -------------------------------------------------------
% Checkmate
% -------------------------------------------------------

is_checkmate(FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    black_king_in_check(FX, FY, TX, TY),
    not bking_can_escape(FX, FY, TX, TY, _, _).

% -------------------------------------------------------
% Output
% -------------------------------------------------------

mate_move(white, queen, FX, FY, TX, TY) :-
    is_checkmate(FX, FY, TX, TY).

#show mate_move/6.