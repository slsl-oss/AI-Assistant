package com.pq.agent_service.controller;

import com.pq.agent_service.domain.dto.UserDTO;
import com.pq.agent_service.domain.vo.LoginVO;
import com.pq.agent_service.domain.vo.UserVO;
import com.pq.agent_service.service.IUserService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@Api(tags = "用户相关接口")
@RestController
@RequestMapping("/users")
@RequiredArgsConstructor
public class UserController {

    private final IUserService userService;

    @PostMapping("/register")
    @ApiOperation("用户注册")
    public UserVO register(@RequestBody UserDTO dto) {
        return userService.register(dto);
    }

    @PostMapping("/login")
    @ApiOperation("用户登录")
    public LoginVO login(@RequestBody UserDTO dto) {
        return userService.login(dto);
    }
}
