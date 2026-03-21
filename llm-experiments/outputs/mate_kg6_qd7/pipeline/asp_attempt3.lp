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
% Queen move generation
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
% Blocking detection for queen moves
% -------------------------------------------------------

blocked_h_r(FX, FY, TX) :-
    piece(white, queen, FX, FY),
    occupied(BX, FY),
    TX > FX, BX > FX, BX < TX.

blocked_h_l(FX, FY, TX) :-
    piece(white, queen, FX, FY),
    occupied(BX, FY),
    TX < FX, BX < FX, BX > TX.

blocked_v_u(FX, FY, TY) :-
    piece(white, queen, FX, FY),
    occupied(FX, BY),
    TY > FY, BY > FY, BY < TY.

blocked_v_d(FX, FY, TY) :-
    piece(white, queen, FX, FY),
    occupied(FX, BY),
    TY < FY, BY < FY, BY > TY.

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
% White king attacks (static - king doesn't move)
% -------------------------------------------------------

wking_attacks(SX, SY) :-
    king(white, KX, KY),
    file(SX), rank(SY),
    SX >= KX - 1, SX =< KX + 1,
    SY >= KY - 1, SY =< KY + 1,
    not same_sq(SX, SY, KX, KY).

% -------------------------------------------------------
% Queen attacks from new position (TX,TY) after move
% -------------------------------------------------------

% Horizontal right from queen's new position
blocked_att_h_r(TX, TY, SX, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX),
    SX > TX,
    occ_after(BX, TY, FX, FY, TX, TY),
    BX > TX, BX < SX.

queen_att_h_r(TX, TY, SX, TY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX),
    SX > TX,
    not blocked_att_h_r(TX, TY, SX, FX, FY).

% Horizontal left from queen's new position
blocked_att_h_l(TX, TY, SX, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX),
    SX < TX,
    occ_after(BX, TY, FX, FY, TX, TY),
    BX < TX, BX > SX.

queen_att_h_l(TX, TY, SX, TY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX),
    SX < TX,
    not blocked_att_h_l(TX, TY, SX, FX, FY).

% Vertical up from queen's new position
blocked_att_v_u(TX, TY, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    rank(SY),
    SY > TY,
    occ_after(TX, BY, FX, FY, TX, TY),
    BY > TY, BY < SY.

queen_att_v_u(TX, TY, TX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    rank(SY),
    SY > TY,
    not blocked_att_v_u(TX, TY, SY, FX, FY).

% Vertical down from queen's new position
blocked_att_v_d(TX, TY, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    rank(SY),
    SY < TY,
    occ_after(TX, BY, FX, FY, TX, TY),
    BY < TY, BY > SY.

queen_att_v_d(TX, TY, TX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    rank(SY),
    SY < TY,
    not blocked_att_v_d(TX, TY, SY, FX, FY).

% Diagonal NE from queen's new position
blocked_att_d_ne(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY > TY,
    SX - TX = SY - TY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX > TX, BX < SX,
    BY > TY, BY < SY,
    BX - TX = BY - TY.

queen_att_d_ne(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY > TY,
    SX - TX = SY - TY,
    not blocked_att_d_ne(TX, TY, SX, SY, FX, FY).

% Diagonal SE from queen's new position
blocked_att_d_se(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY < TY,
    SX - TX = TY - SY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX > TX, BX < SX,
    BY < TY, BY > SY,
    BX - TX = TY - BY.

queen_att_d_se(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY < TY,
    SX - TX = TY - SY,
    not blocked_att_d_se(TX, TY, SX, SY, FX, FY).

% Diagonal NW from queen's new position
blocked_att_d_nw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY > TY,
    TX - SX = SY - TY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX < TX, BX > SX,
    BY > TY, BY < SY,
    TX - BX = BY - TY.

queen_att_d_nw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY > TY,
    TX - SX = SY - TY,
    not blocked_att_d_nw(TX, TY, SX, SY, FX, FY).

% Diagonal SW from queen's new position
blocked_att_d_sw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY < TY,
    TX - SX = TY - SY,
    occ_after(BX, BY, FX, FY, TX, TY),
    BX < TX, BX > SX,
    BY < TY, BY > SY,
    TX - BX = TY - BY.

queen_att_d_sw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY < TY,
    TX - SX = TY - SY,
    not blocked_att_d_sw(TX, TY, SX, SY, FX, FY).

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