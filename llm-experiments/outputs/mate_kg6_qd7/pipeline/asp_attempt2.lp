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
% Queen move generation (horizontal, vertical, diagonal)
% -------------------------------------------------------

queen_move_h_r(FX, FY, TX, FY) :-
    piece(white, queen, FX, FY),
    file(TX),
    TX > FX.
queen_move_h_l(FX, FY, TX, FY) :-
    piece(white, queen, FX, FY),
    file(TX),
    TX < FX.

queen_move_v_u(FX, FY, FX, TY) :-
    piece(white, queen, FX, FY),
    rank(TY),
    TY > FY.
queen_move_v_d(FX, FY, FX, TY) :-
    piece(white, queen, FX, FY),
    rank(TY),
    TY < FY.

queen_move_d_ne(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX > FX, TY > FY,
    TX - FX = TY - FY.
queen_move_d_se(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX > FX, TY < FY,
    TX - FX = FY - TY.
queen_move_d_nw(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX < FX, TY > FY,
    FX - TX = TY - FY.
queen_move_d_sw(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX < FX, TY < FY,
    FX - TX = FY - TY.

% -------------------------------------------------------
% Blocking detection
% -------------------------------------------------------

blocked_h_r(FX, FY, TX) :-
    piece(white, queen, FX, FY),
    occupied(BX, FY),
    FX < TX, BX > FX, BX < TX.
blocked_h_l(FX, FY, TX) :-
    piece(white, queen, FX, FY),
    occupied(BX, FY),
    FX > TX, BX < FX, BX > TX.

blocked_v_u(FX, FY, TY) :-
    piece(white, queen, FX, FY),
    occupied(FX, BY),
    FY < TY, BY > FY, BY < TY.
blocked_v_d(FX, FY, TY) :-
    piece(white, queen, FX, FY),
    occupied(FX, BY),
    FY > TY, BY < FY, BY > TY.

blocked_d_ne(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    occupied(BX, BY),
    TX > FX, TY > FY,
    TX - FX = TY - FY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.
blocked_d_se(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    occupied(BX, BY),
    TX > FX, TY < FY,
    TX - FX = FY - TY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.
blocked_d_nw(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    occupied(BX, BY),
    TX < FX, TY > FY,
    FX - TX = TY - FY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.
blocked_d_sw(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
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
    queen_move_h_r(FX, FY, TX, TY),
    not blocked_h_r(FX, FY, TX),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_h_l(FX, FY, TX, TY),
    not blocked_h_l(FX, FY, TX),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_v_u(FX, FY, TX, TY),
    not blocked_v_u(FX, FY, TY),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_v_d(FX, FY, TX, TY),
    not blocked_v_d(FX, FY, TY),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_d_ne(FX, FY, TX, TY),
    not blocked_d_ne(FX, FY, TX, TY),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_d_se(FX, FY, TX, TY),
    not blocked_d_se(FX, FY, TX, TY),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_d_nw(FX, FY, TX, TY),
    not blocked_d_nw(FX, FY, TX, TY),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_d_sw(FX, FY, TX, TY),
    not blocked_d_sw(FX, FY, TX, TY),
    not occupied_by(white, TX, TY).

% -------------------------------------------------------
% Occupied squares after queen move from (FX,FY) to (TX,TY)
% -------------------------------------------------------

is_from(X, Y, X, Y).

occ_after(TX, TY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY).

occ_after(X, Y, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    occupied(X, Y),
    not is_from(X, Y, FX, FY).

% White occupied after move
occ_after_white(TX, TY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY).

occ_after_white(X, Y, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    occupied_by(white, X, Y),
    not is_from(X, Y, FX, FY).

% -------------------------------------------------------
% White king attacks (static)
% -------------------------------------------------------

wking_attacks(SX, SY) :-
    king(white, KX, KY),
    file(SX), rank(SY),
    SX >= KX - 1, SX =< KX + 1,
    SY >= KY - 1, SY =< KY + 1,
    not is_from(SX, SY, KX, KY).

% -------------------------------------------------------
% Queen attacks from position (TX,TY) after move, on square (SX,SY)
% -------------------------------------------------------

% Horizontal right
queen_att_h_r(TX, TY, SX, TY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX),
    SX > TX,
    not blocked_after_h_r(TX, TY, SX, FX, FY).

blocked_after_h_r(TX, TY, SX, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, TY, FX, FY, TX, TY),
    BX > TX, BX < SX.

% Horizontal left
queen_att_h_l(TX, TY, SX, TY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX),
    SX < TX,
    not blocked_after_h_l(TX, TY, SX, FX, FY).

blocked_after_h_l(TX, TY, SX, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, TY, FX, FY, TX, TY),
    BX < TX, BX > SX.

% Vertical up
queen_att_v_u(TX, TY, TX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    rank(SY),
    SY > TY,
    not blocked_after_v_u(TX, TY, SY, FX, FY).

blocked_after_v_u(TX, TY, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(TX, BY, FX, FY, TX, TY),
    BY > TY, BY < SY.

% Vertical down
queen_att_v_d(TX, TY, TX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    rank(SY),
    SY < TY,
    not blocked_after_v_d(TX, TY, SY, FX, FY).

blocked_after_v_d(TX, TY, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(TX, BY, FX, FY, TX, TY),
    BY < TY, BY > SY.

% Diagonal NE
queen_att_d_ne(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY > TY,
    SX - TX = SY - TY,
    not blocked_after_d_ne(TX, TY, SX, SY, FX, FY).

blocked_after_d_ne(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, BY, FX, FY, TX, TY),
    BX > TX, BX < SX,
    BY > TY, BY < SY,
    BX - TX = BY - TY.

% Diagonal SE
queen_att_d_se(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY < TY,
    SX - TX = TY - SY,
    not blocked_after_d_se(TX, TY, SX, SY, FX, FY).

blocked_after_d_se(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, BY, FX, FY, TX, TY),
    BX > TX, BX < SX,
    BY < TY, BY > SY,
    BX - TX = TY - BY.

% Diagonal NW
queen_att_d_nw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY > TY,
    TX - SX = SY - TY,
    not blocked_after_d_nw(TX, TY, SX, SY, FX, FY).

blocked_after_d_nw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, BY, FX, FY, TX, TY),
    BX < TX, BX > SX,
    BY > TY, BY < SY,
    TX - BX = BY - TY.

% Diagonal SW
queen_att_d_sw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY < TY,
    TX - SX = TY - SY,
    not blocked_after_d_sw(TX, TY, SX, SY, FX, FY).

blocked_after_d_sw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, BY, FX, FY, TX, TY),
    BX < TX, BX > SX,
    BY < TY, BY > SY,
    TX - BX = TY - BY.

% -------------------------------------------------------
% Combined white attacks after queen move
% -------------------------------------------------------

white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_att_h_r(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_att_h_l(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_att_v_u(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_att_v_d(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_att_d_ne(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_att_d_se(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_att_d_nw(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_att_d_sw(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    wking_attacks(SX, SY).

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
    not is_from(EX, EY, KX, KY).

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