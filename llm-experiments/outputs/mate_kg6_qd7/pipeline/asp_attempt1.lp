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

% Horizontal queen move (same rank)
queen_move_h(FX, FY, TX, FY) :-
    piece(white, queen, FX, FY),
    file(TX),
    TX > FX.
queen_move_h(FX, FY, TX, FY) :-
    piece(white, queen, FX, FY),
    file(TX),
    TX < FX.

% Vertical queen move (same file)
queen_move_v(FX, FY, FX, TY) :-
    piece(white, queen, FX, FY),
    rank(TY),
    TY > FY.
queen_move_v(FX, FY, FX, TY) :-
    piece(white, queen, FX, FY),
    rank(TY),
    TY < FY.

% Diagonal queen move
queen_move_d(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX > FX, TY > FY,
    TX - FX = TY - FY.
queen_move_d(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX > FX, TY < FY,
    TX - FX = FY - TY.
queen_move_d(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX < FX, TY > FY,
    FX - TX = TY - FY.
queen_move_d(FX, FY, TX, TY) :-
    piece(white, queen, FX, FY),
    file(TX), rank(TY),
    TX < FX, TY < FY,
    FX - TX = FY - TY.

% -------------------------------------------------------
% Blocking detection
% -------------------------------------------------------

% Horizontal blocking
blocked_h(FX, FY, TX) :-
    occupied(BX, FY),
    piece(white, queen, FX, FY),
    FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    occupied(BX, FY),
    piece(white, queen, FX, FY),
    FX > TX, BX < FX, BX > TX.

% Vertical blocking
blocked_v(FX, FY, TY) :-
    occupied(FX, BY),
    piece(white, queen, FX, FY),
    FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    occupied(FX, BY),
    piece(white, queen, FX, FY),
    FY > TY, BY < FY, BY > TY.

% Diagonal blocking (all four directions)
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    piece(white, queen, FX, FY),
    TX > FX, TY > FY,
    TX - FX = TY - FY,
    BX > FX, BX < TX,
    BY > FY, BY < TY,
    BX - FX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    piece(white, queen, FX, FY),
    TX > FX, TY < FY,
    TX - FX = FY - TY,
    BX > FX, BX < TX,
    BY < FY, BY > TY,
    BX - FX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    piece(white, queen, FX, FY),
    TX < FX, TY > FY,
    FX - TX = TY - FY,
    BX < FX, BX > TX,
    BY > FY, BY < TY,
    FX - BX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    occupied(BX, BY),
    piece(white, queen, FX, FY),
    TX < FX, TY < FY,
    FX - TX = FY - TY,
    BX < FX, BX > TX,
    BY < FY, BY > TY,
    FX - BX = FY - BY.

% -------------------------------------------------------
% Valid queen moves (not blocked, not capturing own piece)
% -------------------------------------------------------

queen_can_move(FX, FY, TX, TY) :-
    queen_move_h(FX, FY, TX, TY),
    not blocked_h(FX, FY, TX),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_v(FX, FY, TX, TY),
    not blocked_v(FX, FY, TY),
    not occupied_by(white, TX, TY).

queen_can_move(FX, FY, TX, TY) :-
    queen_move_d(FX, FY, TX, TY),
    not blocked_d(FX, FY, TX, TY),
    not occupied_by(white, TX, TY).

% -------------------------------------------------------
% After queen moves from (FX,FY) to (TX,TY):
% compute new occupied squares
% -------------------------------------------------------

% Occupied after move (queen moved away from FX,FY to TX,TY)
occ_after(TX, TY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY).

occ_after(X, Y, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    occupied(X, Y),
    not is_from(X, Y, FX, FY).

is_from(X, Y, X, Y).

% -------------------------------------------------------
% Squares attacked by white after queen moves to (TX,TY)
% -------------------------------------------------------

% White king attacks
wking_attacks(KX, KY, SX, SY) :-
    king(white, KX, KY),
    file(SX), rank(SY),
    SX >= KX - 1, SX =< KX + 1,
    SY >= KY - 1, SY =< KY + 1,
    not is_from(SX, SY, KX, KY).

% Queen attacks after move (horizontal)
queen_attacks_h(TX, TY, SX, TY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX),
    SX > TX,
    not blocked_after_h_r(TX, TY, SX, FX, FY).
queen_attacks_h(TX, TY, SX, TY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX),
    SX < TX,
    not blocked_after_h_l(TX, TY, SX, FX, FY).

blocked_after_h_r(TX, TY, SX, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, TY, FX, FY, TX, TY),
    BX > TX, BX < SX.
blocked_after_h_l(TX, TY, SX, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, TY, FX, FY, TX, TY),
    BX < TX, BX > SX.

% Queen attacks after move (vertical)
queen_attacks_v(TX, TY, TX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    rank(SY),
    SY > TY,
    not blocked_after_v_u(TX, TY, SY, FX, FY).
queen_attacks_v(TX, TY, TX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    rank(SY),
    SY < TY,
    not blocked_after_v_d(TX, TY, SY, FX, FY).

blocked_after_v_u(TX, TY, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(TX, BY, FX, FY, TX, TY),
    BY > TY, BY < SY.
blocked_after_v_d(TX, TY, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(TX, BY, FX, FY, TX, TY),
    BY < TY, BY > SY.

% Queen attacks after move (diagonal)
queen_attacks_d(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY > TY,
    SX - TX = SY - TY,
    not blocked_after_d_ne(TX, TY, SX, SY, FX, FY).
queen_attacks_d(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX > TX, SY < TY,
    SX - TX = TY - SY,
    not blocked_after_d_se(TX, TY, SX, SY, FX, FY).
queen_attacks_d(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY > TY,
    TX - SX = SY - TY,
    not blocked_after_d_nw(TX, TY, SX, SY, FX, FY).
queen_attacks_d(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    file(SX), rank(SY),
    SX < TX, SY < TY,
    TX - SX = TY - SY,
    not blocked_after_d_sw(TX, TY, SX, SY, FX, FY).

blocked_after_d_ne(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, BY, FX, FY, TX, TY),
    BX > TX, BX < SX,
    BY > TY, BY < SY,
    BX - TX = BY - TY.
blocked_after_d_se(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, BY, FX, FY, TX, TY),
    BX > TX, BX < SX,
    BY < TY, BY > SY,
    BX - TX = TY - BY.
blocked_after_d_nw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, BY, FX, FY, TX, TY),
    BX < TX, BX > SX,
    BY > TY, BY < SY,
    TX - BX = BY - TY.
blocked_after_d_sw(TX, TY, SX, SY, FX, FY) :-
    queen_can_move(FX, FY, TX, TY),
    occ_after(BX, BY, FX, FY, TX, TY),
    BX < TX, BX > SX,
    BY < TY, BY > SY,
    TX - BX = TY - BY.

% -------------------------------------------------------
% Square attacked by white after queen move
% -------------------------------------------------------

white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_attacks_h(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_attacks_v(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_attacks_d(TX, TY, SX, SY, FX, FY).
white_attacks_after(SX, SY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    wking_attacks(_, _, SX, SY).

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

% Squares the black king can potentially move to
bking_escape_candidate(KX, KY, EX, EY) :-
    king(black, KX, KY),
    file(EX), rank(EY),
    EX >= KX - 1, EX =< KX + 1,
    EY >= KY - 1, EY =< KY + 1,
    not is_from(EX, EY, KX, KY).

% Black king can escape to (EX,EY) after white plays (FX,FY)->(TX,TY) if:
% - not occupied by black piece
% - not attacked by white after the move
% - not occupied by white (except the queen's destination which is now there)

bking_can_escape(FX, FY, TX, TY, EX, EY) :-
    queen_can_move(FX, FY, TX, TY),
    king(black, KX, KY),
    bking_escape_candidate(KX, KY, EX, EY),
    not occupied_by(black, EX, EY),
    not white_attacks_after(EX, EY, FX, FY, TX, TY),
    not occ_after_white(EX, EY, FX, FY, TX, TY).

% White pieces after move (for blocking black king escape)
occ_after_white(TX, TY, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY).
occ_after_white(X, Y, FX, FY, TX, TY) :-
    queen_can_move(FX, FY, TX, TY),
    occupied_by(white, X, Y),
    not is_from(X, Y, FX, FY).

% -------------------------------------------------------
% Checkmate: black king in check and no escape
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