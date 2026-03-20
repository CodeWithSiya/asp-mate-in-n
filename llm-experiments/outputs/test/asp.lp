% Define valid squares
square(X, Y) :- X = 1..8, Y = 1..8.

% Define piece movements
% Rook moves horizontally and vertically
rook_move(X1, Y1, X2, Y2) :- 
    square(X1, Y1), square(X2, Y2),
    (X1 = X2; Y1 = Y2),
    (X1, Y1) != (X2, Y2).

% Queen moves like rook and bishop
queen_move(X1, Y1, X2, Y2) :- rook_move(X1, Y1, X2, Y2).
queen_move(X1, Y1, X2, Y2) :- bishop_move(X1, Y1, X2, Y2).

% Bishop moves diagonally
bishop_move(X1, Y1, X2, Y2) :-
    square(X1, Y1), square(X2, Y2),
    |X1 - X2| = |Y1 - Y2|,
    (X1, Y1) != (X2, Y2).

% Knight moves in L-shape
knight_move(X1, Y1, X2, Y2) :-
    square(X1, Y1), square(X2, Y2),
    ((|X1 - X2| = 2, |Y1 - Y2| = 1) ; (|X1 - X2| = 1, |Y1 - Y2| = 2)).

% King moves one square in any direction
king_move(X1, Y1, X2, Y2) :-
    square(X1, Y1), square(X2, Y2),
    |X1 - X2| <= 1, |Y1 - Y2| <= 1,
    (X1, Y1) != (X2, Y2).

% Pawn moves (simplified - forward one square, captures diagonally)
pawn_move(white, X1, Y1, X2, Y2) :-
    square(X1, Y1), square(X2, Y2),
    X1 = X2, Y2 = Y1 + 1.
pawn_move(black, X1, Y1, X2, Y2) :-
    square(X1, Y1), square(X2, Y2),
    X1 = X2, Y2 = Y1 - 1.

% Pawn captures
pawn_capture(white, X1, Y1, X2, Y2) :-
    square(X1, Y1), square(X2, Y2),
    |X1 - X2| = 1, Y2 = Y1 + 1.
pawn_capture(black, X1, Y1, X2, Y2) :-
    square(X1, Y1), square(X2, Y2),
    |X1 - X2| = 1, Y2 = Y1 - 1.

% Path blocking for sliding pieces
blocked_h(FX, FY, TX) :-
    piece(_, _, BX, FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    piece(_, _, BX, FY), FX > TX, BX < FX, BX > TX.
blocked_h(FX, FY, TX) :-
    king(_, BX, FY), FX < TX, BX > FX, BX < TX.
blocked_h(FX, FY, TX) :-
    king(_, BX, FY), FX > TX, BX < FX, BX > TX.

blocked_v(FX, FY, TY) :-
    piece(_, _, FX, BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    piece(_, _, FX, BY), FY > TY, BY < FY, BY > TY.
blocked_v(FX, FY, TY) :-
    king(_, FX, BY), FY < TY, BY > FY, BY < TY.
blocked_v(FX, FY, TY) :-
    king(_, FX, BY), FY > TY, BY < FY, BY > TY.

blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY), 
    FX < TX, FY < TY, BX > FX, BY > FY, BX < TX, BY < TY,
    BX - FX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX > TX, FY > TY, BX < FX, BY < FY, BX > TX, BY > TY,
    FX - BX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX < TX, FY > TY, BX > FX, BY < FY, BX < TX, BY > TY,
    BX - FX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    piece(_, _, BX, BY),
    FX > TX, FY < TY, BX < FX, BY > FY, BX > TX, BY < TY,
    FX - BX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY), 
    FX < TX, FY < TY, BX > FX, BY > FY, BX < TX, BY < TY,
    BX - FX = BY - FY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX > TX, FY > TY, BX < FX, BY < FY, BX > TX, BY > TY,
    FX - BX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX < TX, FY > TY, BX > FX, BY < FY, BX < TX, BY > TY,
    BX - FX = FY - BY.
blocked_d(FX, FY, TX, TY) :-
    king(_, BX, BY),
    FX > TX, FY < TY, BX < FX, BY > FY, BX > TX, BY < TY,
    FX - BX = BY - FY.

% Valid moves for each piece type
valid_move(Color, rook, FX, FY, TX, TY) :-
    piece(Color, rook, FX, FY),
    rook_move(FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY),
    FX = TX, not blocked_v(FX, FY, TY).
valid_move(Color, rook, FX, FY, TX, TY) :-
    piece(Color, rook, FX, FY),
    rook_move(FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY),
    FY = TY, not blocked_h(FX, FY, TX).

valid_move(Color, queen, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    queen_move(FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY),
    FX = TX, not blocked_v(FX, FY, TY).
valid_move(Color, queen, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    queen_move(FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY),
    FY = TY, not blocked_h(FX, FY, TX).
valid_move(Color, queen, FX, FY, TX, TY) :-
    piece(Color, queen, FX, FY),
    bishop_move(FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY),
    not blocked_d(FX, FY, TX, TY).

valid_move(Color, bishop, FX, FY, TX, TY) :-
    piece(Color, bishop, FX, FY),
    bishop_move(FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY),
    not blocked_d(FX, FY, TX, TY).

valid_move(Color, knight, FX, FY, TX, TY) :-
    piece(Color, knight, FX, FY),
    knight_move(FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY).

valid_move(Color, king, FX, FY, TX, TY) :-
    king(Color, FX, FY),
    king_move(FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY).

valid_move(Color, pawn, FX, FY, TX, TY) :-
    piece(Color, pawn, FX, FY),
    pawn_move(Color, FX, FY, TX, TY),
    not piece(_, _, TX, TY),
    not king(_, TX, TY).

% Captures
valid_capture(Color, Type, FX, FY, TX, TY) :-
    piece(Color, Type, FX, FY),
    Type != pawn,
    valid_move(Color, Type, FX, FY, TX, TY),
    piece(OpColor, _, TX, TY),
    Color != OpColor.
valid_capture(Color, Type, FX, FY, TX, TY) :-
    piece(Color, Type, FX, FY),
    Type != pawn,
    valid_move(Color, Type, FX, FY, TX, TY),
    king(OpColor, TX, TY),
    Color != OpColor.

valid_capture(Color, pawn, FX, FY, TX, TY) :-
    piece(Color, pawn, FX, FY),
    pawn_capture(Color, FX, FY, TX, TY),
    piece(OpColor, _, TX, TY),
    Color != OpColor.
valid_capture(Color, pawn, FX, FY, TX, TY) :-
    piece(Color, pawn, FX, FY),
    pawn_capture(Color, FX, FY, TX, TY),
    king(OpColor, TX, TY),
    Color != OpColor.

% Define opponent
opponent(white, black).
opponent(black, white).

% Check if a move gives check
gives_check(Color, Type, FX, FY, TX, TY) :-
    to_move(Color),
    opponent(Color, OpColor),
    king(OpColor, KX, KY),
    valid_move(Color, Type, FX, FY, TX, TY),
    attacks_square(Color, Type, TX, TY, KX, KY).
gives_check(Color, Type, FX, FY, TX, TY) :-
    to_move(Color),
    opponent(Color, OpColor),
    king(OpColor, KX, KY),
    valid_capture(Color, Type, FX, FY, TX, TY),
    attacks_square(Color, Type, TX, TY, KX, KY).

% Define attacks after a move
attacks_square(Color, rook, TX, TY, KX, KY) :-
    rook_move(TX, TY, KX, KY),
    TX = KX, not blocked_v_after_move(TX, TY, KY, TX, TY).
attacks_square(Color, rook, TX, TY, KX, KY) :-
    rook_move(TX, TY, KX, KY),
    TY = KY, not blocked_h_after_move(TX, TY, KX, TX, TY).

attacks_square(Color, queen, TX, TY, KX, KY) :-
    attacks_square(Color, rook, TX, TY, KX, KY).
attacks_square(Color, queen, TX, TY, KX, KY) :-
    attacks_square(Color, bishop, TX, TY, KX, KY).

attacks_square(Color, bishop, TX, TY, KX, KY) :-
    bishop_move(TX, TY, KX, KY),
    not blocked_d_after_move(TX, TY, KX, KY, TX, TY).

attacks_square(Color, knight, TX, TY, KX, KY) :-
    knight_move(TX, TY, KX, KY).

attacks_square(Color, king, TX, TY, KX, KY) :-
    king_move(TX, TY, KX, KY).

attacks_square(white, pawn, TX, TY, KX, KY) :-
    pawn_capture(white, TX, TY, KX, KY).
attacks_square(black, pawn, TX, TY, KX, KY) :-
    pawn_capture(black, TX, TY, KX, KY).

% Blocking after move (considering the piece has moved)
blocked_h_after_move(FX, FY, TX, MX, MY) :-
    piece(_, _, BX, FY), BX != MX, FX < TX, BX > FX, BX < TX.
blocked_h_after_move(FX, FY, TX, MX, MY) :-
    piece(_, _, BX, FY), BX != MX, FX > TX, BX < FX, BX > TX.
blocked_h_after_move(FX, FY, TX, MX, MY) :-
    king(_, BX, FY), (BX, FY) != (MX, MY), FX < TX, BX > FX, BX < TX.
blocked_h_after_move(FX, FY, TX, MX, MY) :-
    king(_, BX, FY), (BX, FY) != (MX, MY), FX > TX, BX < FX, BX > TX.

blocked_v_after_move(FX, FY, TY, MX, MY) :-
    piece(_, _, FX, BY), BY != MY, FY < TY, BY > FY, BY < TY.
blocked_v_after_move(FX, FY, TY, MX, MY) :-
    piece(_, _, FX, BY), BY != MY, FY > TY, BY < FY, BY > TY.
blocked_v_after_move(FX, FY, TY, MX, MY) :-
    king(_, FX, BY), (FX, BY) != (MX, MY), FY < TY, BY > FY, BY < TY.
blocked_v_after_move(FX, FY, TY, MX, MY) :-
    king(_, FX, BY), (FX, BY) != (MX, MY), FY > TY, BY < FY, BY > TY.

blocked_d_after_move(FX, FY, TX, TY, MX, MY) :-
    piece(_, _, BX, BY), (BX, BY) != (MX, MY),
    FX < TX, FY < TY, BX > FX, BY > FY, BX < TX, BY < TY,
    BX - FX = BY - FY.
blocked_d_after_move(FX, FY, TX, TY, MX, MY) :-
    piece(_, _, BX, BY), (BX, BY) != (MX, MY),
    FX > TX, FY > TY, BX < FX, BY < FY, BX > TX, BY > TY,
    FX - BX = FY - BY.
blocked_d_after_move(FX, FY, TX, TY, MX, MY) :-
    piece(_, _, BX, BY), (BX, BY) != (MX, MY),
    FX < TX, FY > TY, BX > FX, BY < FY, BX < TX, BY > TY,
    BX - FX = FY - BY.
blocked_d_after_move(FX, FY, TX, TY, MX, MY) :-
    piece(_, _, BX, BY), (BX, BY) != (MX, MY),
    FX > TX, FY < TY, BX < FX, BY > FY, BX > TX, BY < TY,
    FX - BX = BY - FY.
blocked_d_after_move(FX, FY, TX, TY, MX, MY) :-
    king(_, BX, BY), (BX, BY) != (MX, MY),
    FX < TX, FY < TY, BX > FX, BY > FY, BX < TX, BY < TY,
    BX - FX = BY - FY.
blocked_d_after_move(FX, FY, TX, TY, MX, MY) :-
    king(_, BX, BY), (BX, BY) != (MX, MY),
    FX > TX, FY > TY, BX < FX, BY < FY, BX > TX, BY > TY,
    FX - BX = FY - BY.
blocked_d_after_move(FX, FY, TX, TY, MX, MY) :-
    king(_, BX, BY), (BX, BY) != (MX, MY),
    FX < TX, FY > TY, BX > FX, BY < FY, BX < TX, BY > TY,
    BX - FX = FY - BY.
blocked_d_after_move(FX, FY, TX, TY, MX, MY) :-
    king(_, BX, BY), (BX, BY) != (MX, MY),
    FX > TX, FY < TY, BX < FX, BY > FY, BX > TX, BY < TY,
    FX - BX = BY - FY.

% King escape squares
king_escape(OpColor, EX, EY) :-
    opponent(Color, OpColor),
    to_move(Color),
    king(OpColor, KX, KY),
    king_move(KX, KY, EX, EY),
    not piece(_, _, EX, EY),
    not king(_, EX, EY).

% Check if escape square is safe after a move
escape_safe(Color, Type, FX, FY, TX, TY, EX, EY) :-
    opponent(Color, OpColor),
    king_escape(OpColor, EX, EY),
    not attacked_after_move(Color, EX, EY, Type, FX, FY, TX, TY).

% Check if a square is attacked after a move
attacked_after_move(Color, EX, EY, MoveType, MFX, MFY, MTX, MTY) :-
    piece(Color, Type, PX, PY),
    (Type, PX, PY) != (MoveType, MFX, MFY),
    attacks_square(Color, Type, PX, PY, EX, EY).
attacked_after_move(Color, EX, EY, MoveType, MFX, MFY, MTX, MTY) :-
    king(Color, KX, KY),
    attacks_square(Color, king, KX, KY, EX, EY).
attacked_after_move(Color, EX, EY, MoveType, MFX, MFY, MTX, MTY) :-
    attacks_square(Color, MoveType, MTX, MTY, EX, EY).

% Check if all escape squares are unsafe
all_escapes_unsafe(Color, Type, FX, FY, TX, TY) :-
    gives_check(Color, Type, FX, FY, TX, TY),
    opponent(Color, OpColor),
    king(OpColor, KX, KY),
    not escape_safe(Color, Type, FX, FY, TX, TY, _, _).

% No interposition possible (simplified - no pieces can block)
no_interposition(Color, Type, FX, FY, TX, TY) :-
    gives_check(Color, Type, FX, FY, TX, TY),
    opponent(Color, OpColor),
    not can_interpose(OpColor, Color, Type, FX, FY, TX, TY).

can_interpose(OpColor, Color, Type, FX, FY, TX, TY) :-
    piece(OpColor, _, PX, PY),
    square(IX, IY),
    valid_move(OpColor, _, PX, PY, IX, IY),
    blocks_check(IX, IY, Color, Type, TX, TY).

blocks_check(IX, IY, Color, rook, TX, TY) :-
    king(OpColor, KX, KY),
    opponent(Color, OpColor),
    TX = KX, TY < KY, IX = TX, IY > TY, IY < KY.
blocks_check(IX, IY, Color, rook, TX, TY) :-
    king(OpColor, KX, KY),
    opponent(Color, OpColor),
    TX = KX, TY > KY, IX = TX, IY < TY, IY > KY.
blocks_check(IX, IY, Color, rook, TX, TY) :-
    king(OpColor, KX, KY),
    opponent(Color, OpColor),
    TY = KY, TX < KX, IY = TY, IX > TX, IX < KX.
blocks_check(IX, IY, Color, rook, TX, TY) :-
    king(OpColor, KX, KY),
    opponent(Color, OpColor),
    TY = KY, TX > KX, IY = TY, IX < TX, IX > KX.

% No capture of attacking piece
no_capture_attacker(Color, Type, FX, FY, TX, TY) :-
    gives_check(Color, Type, FX, FY, TX, TY),
    opponent(Color, OpColor),
    not can_capture_attacker(OpColor, TX, TY).

can_capture_attacker(OpColor, TX, TY) :-
    piece(OpColor, _, PX, PY),
    valid_capture(OpColor, _, PX, PY, TX, TY).
can_capture_attacker(OpColor, TX, TY) :-
    king(OpColor, KX, KY),
    king_move(KX, KY, TX, TY).

% Mate condition
mate_move(Color, Type, FX, FY, TX, TY) :-
    to_move(Color),
    gives_check(Color, Type, FX, FY, TX, TY),
    all_escapes_unsafe(Color, Type, FX, FY, TX, TY),
    no_interposition(Color, Type, FX, FY, TX, TY),
    no_capture_attacker(Color, Type, FX, FY, TX, TY).

#show mate_move/6.